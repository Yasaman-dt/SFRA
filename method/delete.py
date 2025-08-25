import copy
import torch
from torch import nn
import torch.nn.functional as F
import itertools
from collections import OrderedDict

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm



@timer
def delete(   ori_model, train_forget_loader,
                    unlearn_epoch, unlearn_rate,
                    results_csv, forget_class,
                    logger, console_handler,
                    loader_dict, experiment_path,
                    soft_label,
                    eval_opt = eval_opt, disable_bn = False,
                    early_stop_patience=30,
                    ):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    
    _, Aor = test(ori_model, loader_dict["test_remain"])
    best_aus = float("-inf")
    best_path = experiment_path / f"lr{unlearn_rate}/ckpt_best_by_aus.pth"
    best_state = None
    epochs_since_improve = 0

    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    test_model = copy.deepcopy(ori_model).to("cuda")

    criterion = nn.KLDivLoss(reduction='batchmean') 

    assert soft_label in ["inf"]
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
            unlearn_model.train()

            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval() 

            unlearn_model.zero_grad()
            optimizer.zero_grad()   

            test_model.eval()
            batch_size = x.shape[0]
            with torch.no_grad():
                pred_label = test_model(x)
            if soft_label == "inf":
                pred_label[torch.arange(batch_size), y] = -1e10
            else:
                raise ValueError("Unknown soft label method")

            ori_logits = unlearn_model(x)

            ori_logits = F.log_softmax(ori_logits, dim=1)   # input log softmax
            pred_label = F.softmax(pred_label, dim=1)       # target softmax
            loss = criterion(ori_logits, pred_label)
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
