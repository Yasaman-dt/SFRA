import copy
import torch
from torch import nn
from torchvision import datasets
from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders

from utils import *
from trainer import *
import log_utils
import tqdm

def noise_label(label, num_classes, approx_different):

    is_label_lst = isinstance(label, list)
    label = torch.tensor(label) if is_label_lst else label

    if approx_different:
        noisy_label = torch.randint(0, num_classes, (len(label),)).long()
    else:
        shift = torch.randint(1, num_classes, (len(label),)).long()
        noisy_label = (label+shift)%num_classes


    return noisy_label.tolist() if is_label_lst else noisy_label    

@timer
def random_label(   ori_model, train_forget_loader, num_classes,
                    unlearn_epoch, unlearn_rate,
                    results_csv, forget_class,
                    fixed_noise_label, 
                    logger, console_handler,
                    loader_dict, experiment_path,
                    approx_different = True, eval_opt = eval_opt, disable_bn=False
                    ):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    
    _, Aor = test(ori_model, loader_dict["test_remain"])
    best_aus = float("-inf")
    best_path = experiment_path / "ckpt_best_by_aus.pth"        
    best_state = None
    
    unlearn_model = copy.deepcopy(ori_model).to("cuda")

    train_forget_loader_randlabel = copy.deepcopy(train_forget_loader)    
    
    if approx_different:
        note_print("")
    else:
        note_print("")

    if fixed_noise_label:   
        if isinstance(train_forget_loader_randlabel.dataset, datasets.ImageFolder):
            note_print(f"imagefolder dataset: type(samples){type(train_forget_loader_randlabel.dataset.samples)}")
            paths, targets = zip(*train_forget_loader_randlabel.dataset.samples) 
            paths, targets = list(paths), list(targets)
            targets = noise_label(targets, num_classes, approx_different)
            train_forget_loader_randlabel.dataset.samples = list(zip(paths, targets))
        elif    isinstance(train_forget_loader_randlabel.dataset, datasets.CIFAR10)\
                or isinstance(train_forget_loader_randlabel.dataset, datasets.CIFAR100)\
                or isinstance(train_forget_loader_randlabel.dataset, vggface_dataset):
            train_forget_loader_randlabel.dataset.targets = noise_label(train_forget_loader_randlabel.dataset.targets, num_classes, approx_different)

    # print(train_forget_loader.dataset.targets == train_forget_loader_randlabel.dataset.targets)    # False
    # batches_per_epoch = len(train_forget_loader_randlabel)
    # len(loader)batch，loader.dataset.data.shape

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
        for x, y in train_forget_loader_randlabel:
        # for x, y in tqdm.tqdm(train_forget_loader_randlabel):
            if not fixed_noise_label:
                raise NotImplementedError("弃用")
                y = noise_label(y, num_classes, approx_different)

            x, y = x.to("cuda"), y.to("cuda")

            unlearn_model.train()
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval() 
            unlearn_model.zero_grad()
            optimizer.zero_grad()   

            logits = unlearn_model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)
    
        _, a_forget = test(unlearn_model, loader_dict["test_forget"])
        _, a_retain = test(unlearn_model, loader_dict["test_remain"])
        aus = calculate_AUS(a_forget, a_retain, Aor)
        if aus > best_aus:
            best_aus = aus
            best_state = {k: v.detach().cpu().clone() for k, v in unlearn_model.state_dict().items()}
            torch.save(unlearn_model, best_path)
            logger.info(f"[epoch {epoch+1}] ★ New best AUS={aus:.4f} (forget={a_forget:.4f}, retain={a_retain:.4f}) -> {best_path}")
        else:
            logger.info(f"[epoch {epoch+1}] AUS={aus:.4f} (forget={a_forget:.4f}, retain={a_retain:.4f})")

    if best_state is not None:
        unlearn_model.load_state_dict(best_state)
        unlearn_model.to("cuda").eval()


    gather_and_write_metrics_csv(
        csv_path=str(results_csv),
        model=unlearn_model,
        method="boundary_shrink",
        forget_class=forget_class,                   
        train_retain_loader=loader_dict["train_remain"],
        train_forget_loader=loader_dict["train_forget"],
        test_retain_loader=loader_dict["test_remain"],
        test_forget_loader=loader_dict["test_forget"],
        train_full_loader=loader_dict.get("train"),   
        test_full_loader=loader_dict.get("test"),    
        mia_result=None,
    )        

    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model