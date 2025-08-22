import copy
import torch
from torch import nn

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


def gradient_saliency_mask(model, forget_loader, threshold_ratio):
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(reduction='sum') 

    model.eval()

    gradients = {}
    for name, param in model.named_parameters():
        # gradients[name] = 0 
        gradients[name] = torch.zeros_like(param)

    for image, target in tqdm.tqdm(forget_loader, desc="生成mask"):
        image = image.cuda()
        target = target.cuda()

        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad(): 
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data
                else:
                    print(f"{name} has no grad")

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])


    # sorted_dict_positions = {} 

    mask_dict = {}  

    # Concatenate all tensors into a single tensor
    all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()]) 

    # Calculate the threshold index for the top 10% elements
    threshold_index = int(len(all_elements) * threshold_ratio)    # len(a)  a.shape[0]

    # Calculate positions of all elements
    positions = torch.argsort(all_elements) 
    ranks = torch.argsort(positions)        


    start_index = 0
    for key, tensor in gradients.items():
        num_elements = tensor.numel()  
        # tensor_positions = positions[start_index: start_index + num_elements]
        tensor_ranks = ranks[start_index : start_index + num_elements]

        # sorted_positions = tensor_ranks.reshape(tensor.shape)
        # sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1    
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        mask_dict[key] = threshold_tensor
        start_index += num_elements
    return mask_dict


@timer
def salun(ori_model, train_forget_loader, num_classes,
                    unlearn_epoch, unlearn_rate,
                    results_csv, forget_class,
                    fixed_noise_label, 
                    logger, console_handler,
                    loader_dict, experiment_path,
                    eval_opt = eval_opt,
                    threshold_ratio=0.1,
                    approx_different = True,
                    retain_data=False,
                    mask=True,
                    disable_bn = False, 
                    ):
    mask_dict = gradient_saliency_mask(ori_model, train_forget_loader, threshold_ratio)
    # mask_dict_v2 = gradient_saliency_mask(ori_model, train_forget_loader, threshold_ratio, negative_loss=True)
    # for _, mask in mask_dict.items():
    #     mask_v2 = mask_dict_v2[_]
    #     print(_, torch.sum(mask != mask_v2))
    #     assert  torch.sum(mask != mask_v2)<10, f"{_} param mask: {torch.sum(mask.float())}!={torch.sum(mask_v2.float())}"

    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    
    _, Aor = test(ori_model, loader_dict["test_remain"])
    best_aus = float("-inf")
    best_path = experiment_path / "ckpt_best_by_aus.pth"        
    best_state = None

    # test_model = copy.deepcopy(ori_model).to("cuda")    # random label
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
            if not fixed_noise_label:
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
            if mask:
                for name, param in unlearn_model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask_dict[name]
            optimizer.step()

        if retain_data: # follow this implementation from https://github.com/OPTML-Group/Unlearn-Saliency/blob/master/Classification/unlearn/RL.py
            for x, y in loader_dict["train_remain"]:
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
                if mask:
                    for name, param in unlearn_model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask_dict[name]
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