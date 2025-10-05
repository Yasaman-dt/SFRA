import copy
import torch
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm
import itertools


#from advertorch.attacks import L2PGDAttack

import torch
import torch.nn.functional as F

class L2PGDAttack:
    def __init__(self, model, eps=0.4, eps_iter=0.1, nb_iter=10,
                 rand_init=True, targeted=True, clip_min=None, clip_max=None):
        self.model = model
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max

    @torch.no_grad()
    def _project(self, delta):
        # per-sample L2 projection to radius eps
        b = delta.shape[0]
        flat = delta.view(b, -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        factors = (self.eps / norms).clamp(max=1.0)
        return (flat * factors).view_as(delta)

    def perturb(self, x, y_target):
        # Work with grads; we’ll turn them on manually
        x_orig = x.detach()
        if self.rand_init:
            delta = torch.randn_like(x_orig)
            delta = self._project(delta)
        else:
            delta = torch.zeros_like(x_orig)

        delta.requires_grad_(True)

        for _ in range(self.nb_iter):
            x_adv = x_orig + delta
            if self.clip_min is not None and self.clip_max is not None:
                x_adv = x_adv.clamp_(self.clip_min, self.clip_max)

            # forward with grads
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y_target)

            # targeted attack minimizes CE toward target;
            # untargeted maximizes CE => flip sign
            grad_sign = 1.0 if self.targeted else -1.0

            # compute grad wrt x_adv
            grad = torch.autograd.grad(grad_sign * loss, x_adv, retain_graph=False, create_graph=False)[0]

            # normalized L2 step
            b = grad.shape[0]
            gflat = grad.view(b, -1)
            gnorm = gflat.norm(p=2, dim=1).view(b, *([1] * (grad.dim() - 1))).clamp(min=1e-12)
            step = grad / gnorm
            with torch.no_grad():
                delta -= self.eps_iter * step
                # stay in L2 ball
                delta = (x_orig + self._project(delta)) - x_orig
            delta.requires_grad_(True)

        x_adv = (x_orig + delta).detach()
        if self.clip_min is not None and self.clip_max is not None:
            x_adv = x_adv.clamp_(self.clip_min, self.clip_max)
        return x_adv


def estimate_parameter_importance(data_loader, model, num_samples=0):  
    importance = {n: torch.zeros(p.shape).to("cuda") for n, p in model.named_parameters()
                  if p.requires_grad}  
    
    n_samples_batches = ((num_samples+1) // data_loader.batch_size) if num_samples > 0 \
        else len(data_loader)
        # else (len(data_loader.dataset) // data_loader.batch_size)   
    
    model.eval()
    for images, targets in itertools.islice(data_loader, n_samples_batches): 
        outputs = model.forward(images.to("cuda"))
        loss = torch.norm(outputs, p=2, dim=1).mean()  
        # optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():  
            for n, p in model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(targets)

    n_samples = n_samples_batches * data_loader.batch_size   
    importance = {n: (p / n_samples) for n, p in importance.items()}
    return importance

def adv_attack(model, adversary, data, target, num_classes):
    model.eval()
    
    data, target = data.to("cuda"), target.to("cuda")

    attack_label = torch.rand(data.shape[0]).cuda() * num_classes
    attack_label = attack_label.to(torch.long)
    # attack_label = torch.where(attack_label == target, (torch.rand(data.shape[0]).long().cuda()*num_classes + num_classes) // 2, attack_label)
    # maybe erro in source code https://github.com/csm9493/L2UL

    adv_example = adversary.perturb(data, attack_label)

    # inputs_numpy = adv_example.detach().cpu().numpy()
    # labels_numpy = attack_label.cpu().numpy()
    inputs_numpy = adv_example.detach()
    labels_numpy = attack_label

    return inputs_numpy, labels_numpy

@timer
def l2ul_adv(ori_model, train_forget_loader, num_classes,
                    unlearn_epoch, unlearn_rate,
                    results_csv, forget_class,
                    logger, console_handler,
                    loader_dict, experiment_path,
                    eval_opt = eval_opt,
                    adv_eps=0.4, adv_lambda=0.1, reg_lambda=0, disable_bn=False,
                    early_stop_patience=30,
                    ):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    
    _, Aor = test(ori_model, loader_dict["test_remain"])
    best_aus = float("-inf")
    best_path = experiment_path / f"lr{unlearn_rate}/ckpt_best_by_aus.pth"        
    best_state = None
    epochs_since_improve = 0

    test_model = copy.deepcopy(ori_model).to("cuda")
    unlearn_model = copy.deepcopy(ori_model).to("cuda")

    if reg_lambda != 0:
        origin_params = {n: p.clone().detach() for n, p in unlearn_model.named_parameters() if p.requires_grad}
        importance = estimate_parameter_importance(train_forget_loader, test_model)
        for key in importance.keys():
            importance[key] = (importance[key] - importance[key].min()) / (importance[key].max() - importance[key].min())  # 对重要性进行归一化
            importance[key] = (1 - importance[key])

    test_model.eval()
    adversary = L2PGDAttack(test_model, eps=adv_eps, eps_iter=0.1, nb_iter=10, rand_init=True, targeted=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)

    accs_dict = {
        'train_forget': [],
        'train_remain': [],
        'test_forget': [],
        'test_remain': []
    }

    log_utils.enable_console_logging(logger, console_handler, False)
    for epoch in tqdm.trange(unlearn_epoch):
        for x, y in train_forget_loader:
            x, y = x.to("cuda"), y.to("cuda")
            test_model.eval()
            x_adv, y_adv = adv_attack(test_model, adversary, x, y, num_classes) # 为3, 224, 224

            unlearn_model.train()
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            unlearn_model.zero_grad()
            optimizer.zero_grad()

            logits = unlearn_model(x)
            unlearn_model.eval()
            logits_adv = unlearn_model(x_adv)

            # loss = -criterion(logits, y)
            loss = -criterion(logits, y) + criterion(logits_adv, y_adv)*adv_lambda
            if reg_lambda != 0:
                loss_reg = 0
                for n, p in unlearn_model.named_parameters():
                    if n in importance.keys():
                        loss_reg += torch.sum(importance[n] * (p - origin_params[n]).pow(2)) / 2
                logger.info(f"loss: {loss.item()}, loss_reg: {loss_reg.item()}")
                # print(f"loss: {loss.item()}, loss_reg: {loss_reg.item()}"[])
                loss += loss_reg * reg_lambda

            loss.backward()
            optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

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

    if best_state is not None:
        unlearn_model.load_state_dict(best_state)
        unlearn_model.to("cuda").eval()

    # gather_and_write_metrics_csv(
    #     csv_path=str(results_csv),
    #     model=unlearn_model,
    #     method="boundary_shrink",
    #     forget_class=forget_class,                   
    #     train_retain_loader=loader_dict["train_remain"],
    #     train_forget_loader=loader_dict["train_forget"],
    #     test_retain_loader=loader_dict["test_remain"],
    #     test_forget_loader=loader_dict["test_forget"],
    #     train_full_loader=loader_dict.get("train"),   
    #     test_full_loader=loader_dict.get("test"),    
    #     mia_result=None,
    # )        
   
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model