# neggrad_plus.py
import copy, torch, tqdm
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils

def l1_regularization(model):
    params = [p.view(-1) for p in model.parameters()]
    return torch.linalg.norm(torch.cat(params), ord=1)

@timer
def neggrad_plus(
    ori_model,
    train_remain_loader,        # D_r
    train_forget_loader,        # D_f
    unlearn_epoch,
    unlearn_rate,
    results_csv,
    forget_class,
    logger, console_handler,
    loader_dict, experiment_path,
    eval_opt = eval_opt,
    # weighting / regularization
    retain_weight: float = 1.0,
    forget_weight: float = 1.0,
    with_l1: bool = False,
    no_l1_epochs: float = float("inf"),
    alpha: float = 0.0,
    # training knobs
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    disable_bn: bool = False,
    early_stop_patience: int = 30,
):
    """
    NegGrad+ : joint objective per step
        L = retain_weight * CE(x_r, y_r)  -  forget_weight * CE(x_f, y_f)  +  (optional L1)
    """
    logger.info(f"[NegGrad+] epochs={unlearn_epoch}, lr={unlearn_rate}, "
                f"w_r={retain_weight}, w_f={forget_weight}, L1={with_l1} (alpha={alpha})")
    logger.info(f"eval option {eval_opt}")

    # For AUS baseline (A_or)
    _, Aor = test(ori_model, loader_dict["test_remain"])

    best_aus = float("-inf")
    best_state = None
    epochs_since_improve = 0
    save_dir = experiment_path / f"lr{unlearn_rate}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "ckpt_best_by_aus.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = copy.deepcopy(ori_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=unlearn_rate, momentum=momentum, weight_decay=weight_decay
    )

    accs_history = {k: [] for k in keys}

    # handy re-iterators for uneven loader lengths
    def make_infinite(loader):
        while True:
            for batch in loader:
                yield batch

    log_utils.enable_console_logging(logger, console_handler, False)

    for epoch in tqdm.trange(unlearn_epoch):
        model.train()
        if disable_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        inf_remain = make_infinite(train_remain_loader)
        inf_forget = make_infinite(train_forget_loader)
        
        # Iterate by the longer of the two loaders
        steps = max(len(train_remain_loader), len(train_forget_loader))
        for _ in range(steps):
            (xr, yr) = next(inf_remain)
            (xf, yf) = next(inf_forget)
            xr, yr, xf, yf = xr.to(device), yr.to(device), xf.to(device), yf.to(device)

            optimizer.zero_grad()

            # forward on retain
            logits_r = model(xr)
            loss_r = criterion(logits_r, yr)

            # forward on forget
            logits_f = model(xf)
            loss_f = criterion(logits_f, yf)

            # NegGrad+: subtract forget term
            loss = retain_weight * loss_r - forget_weight * loss_f

            # Optional scheduled L1 like your other methods
            if with_l1:
                if epoch < unlearn_epoch - no_l1_epochs:
                    cur_alpha = alpha * (1 - epoch / (unlearn_epoch - no_l1_epochs))
                else:
                    cur_alpha = 0.0
                loss = loss + cur_alpha * l1_regularization(model)

            loss.backward()
            optimizer.step()

        logger.info(f"[epoch {epoch+1}] loss_r={loss_r.item():.4f} loss_f={loss_f.item():.4f} "
                    f"obj={loss.item():.4f}")

        # eval + plots (same flow as your other methods)
        cur = evaluate_model_on_all_loaders(model, loader_dict, eval_opt, logger)
        for k in keys: accs_history[k].append(cur[k])
        plot_unlearn_remain_acc_figure(epoch+1, accs_history, save_dir)

        # track AUS
        _, a_forget = test(model, loader_dict["test_forget"])
        _, a_retain = test(model, loader_dict["test_remain"])
        aus = calculate_AUS(a_forget, a_retain, Aor)

        if aus > best_aus:
            best_aus = aus
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model, best_path)
            logger.info(f"[epoch {epoch+1}] ★ New best AUS={aus:.4f} "
                        f"(forget={a_forget:.4f}, retain={a_retain:.4f}) -> {best_path}")
            epochs_since_improve = 0
        else:
            logger.info(f"[epoch {epoch+1}] AUS={aus:.4f} "
                        f"(forget={a_forget:.4f}, retain={a_retain:.4f})")
            epochs_since_improve += 1
            logger.info(f"[epoch {epoch+1}] early-stop patience {epochs_since_improve}/{early_stop_patience}")
            if epochs_since_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device).eval()

    log_utils.enable_console_logging(logger, console_handler, True)
    return model
