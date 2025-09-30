import copy, tqdm, torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils

class KLTeacherToStudent(nn.Module):
    """D_KL( p_teacher || p_student ) with temperature."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.T = temperature
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def forward(self, teacher_logits, student_logits):
        T = self.T
        with torch.no_grad():
            p_t = F.softmax(teacher_logits / T, dim=-1)
        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        return self.kldiv(log_p_s, p_t) * (T * T)

def scrub(
    ori_model,
    train_forget_loader,          # D_f (max phase)
    num_classes,
    unlearn_epoch,
    unlearn_rate,
    results_csv,
    forget_class,
    logger, console_handler,
    loader_dict,                  # needs: "train_remain","test_forget","test_remain"
    experiment_path,
    eval_opt = eval_opt, 
    # ---- SCRUB knobs ----
    alpha=1.0,                    # weight on KL retain
    gamma=1.0,                    # weight on CE retain
    temperature=1.0,
    extra_min_epochs=0,           # extra retain-only epochs after each max
    retain_data=True,
    # ---- early stopping / rewind ----
    early_stop_patience=30,
    do_rewind=True,
    val_forget_like_loader=None,  # optional; enables SCRUB+R
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # frozen teacher
    teacher = copy.deepcopy(ori_model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # student to unlearn
    unlearn_model = copy.deepcopy(ori_model).to(device)

    logger.info(f"[SCRUB] epochs={unlearn_epoch}, lr={unlearn_rate}, alpha={alpha}, gamma={gamma}, T={temperature}")
    logger.info(f"eval option {eval_opt}")

    _, Aor = test(ori_model, loader_dict["test_remain"])

    kl_t2s = KLTeacherToStudent(temperature=temperature)
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)

    accs_dict = { 'train_forget': [], 'train_remain': [], 'test_forget': [], 'test_remain': [] }

    best_aus = float("-inf")
    best_path = experiment_path / f"lr{unlearn_rate}/ckpt_best_by_aus.pth"
    best_state = None
    epochs_since_improve = 0

    checkpoints = []  # (epoch, state_dict_cpu, forget_err)

    log_utils.enable_console_logging(logger, console_handler, False)

    for epoch in tqdm.trange(unlearn_epoch, desc="SCRUB"):
        # ===== MAX on D_f: ascend KL (teacher || student) =====
        unlearn_model.train()
        for x_f, _ in train_forget_loader:
            x_f = x_f.to(device)
            with torch.no_grad():
                t_logits = teacher(x_f)
            s_logits = unlearn_model(x_f)
            loss_max = - kl_t2s(t_logits, s_logits)  # ascend KL via negative sign
            optimizer.zero_grad(set_to_none=True)
            loss_max.backward()
            optimizer.step()

        # ===== MIN on D_r: minimize alpha*KL + gamma*CE =====
        min_rounds = 1 + int(extra_min_epochs)
        if retain_data:
            for _ in range(min_rounds):
                unlearn_model.train()
                for x_r, y_r in loader_dict["train_remain"]:
                    x_r, y_r = x_r.to(device), y_r.to(device)
                    with torch.no_grad():
                        t_logits_r = teacher(x_r)
                    s_logits_r = unlearn_model(x_r)
                    loss_min = alpha * kl_t2s(t_logits_r, s_logits_r) + gamma * ce(s_logits_r, y_r)
                    optimizer.zero_grad(set_to_none=True)
                    loss_min.backward()
                    optimizer.step()

        scalar_loss = (loss_min if retain_data else loss_max).item()
        logger.info(f"[epoch {epoch+1}] loss {scalar_loss:.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for k in accs_dict.keys():
            accs_dict[k].append(cur_accs_dict[k])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path/f"lr{unlearn_rate}")

        _, a_forget = test(unlearn_model, loader_dict["test_forget"])
        _, a_retain = test(unlearn_model, loader_dict["test_remain"])
        aus = calculate_AUS(a_forget, a_retain, Aor)

        if aus > best_aus:
            best_aus = aus
            best_state = {k: v.detach().cpu().clone() for k, v in unlearn_model.state_dict().items()}
            torch.save(unlearn_model, best_path)
            logger.info(f"[epoch {epoch+1}] ★ New best AUS={aus:.4f} (forget={a_forget:.4f}, retain={a_retain:.4f}) -> {best_path}")
            epochs_since_improve = 0
        else:
            logger.info(f"[epoch {epoch+1}] AUS={aus:.4f} (forget={a_forget:.4f}, retain={a_retain:.4f})")
            epochs_since_improve += 1
            logger.info(f"[epoch {epoch+1}] early-stop patience {epochs_since_improve}/{early_stop_patience}")
            if epochs_since_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}: AUS hasn't improved for {early_stop_patience} epochs.")
                break

        # keep checkpoint for optional rewind (use forget error)
        forget_err = max(0.0, 100.0 - a_forget)
        checkpoints.append((epoch+1, {k: v.detach().cpu().clone() for k, v in unlearn_model.state_dict().items()}, forget_err))

    if best_state is not None:
        unlearn_model.load_state_dict(best_state)
        unlearn_model.to(device).eval()

    # ===== Optional rewind (SCRUB+R) =====
    if do_rewind and val_forget_like_loader is not None and len(checkpoints) > 0:
        logger.info("[SCRUB+R] Computing reference forget error...")
        def _err_on(loader):
            _, acc = test(unlearn_model, loader)
            return max(0.0, 100.0 - acc)

        ref_err = _err_on(val_forget_like_loader)
        idx = min(range(len(checkpoints)), key=lambda i: abs(checkpoints[i][2] - ref_err))
        chosen_epoch, chosen_state, chosen_forget_err = checkpoints[idx]
        unlearn_model.load_state_dict(chosen_state)
        unlearn_model.to(device).eval()
        logger.info(f"[SCRUB+R] rewound to epoch {chosen_epoch} (forget_err={chosen_forget_err:.4f})")

    log_utils.enable_console_logging(logger, console_handler, True)
    return unlearn_model
