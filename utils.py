import os, sys
import random
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import  transforms, datasets
from torch import nn
import numpy as np
import time
from functools import wraps
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import csv, json

@torch.inference_mode()
def _infer_num_classes(model, any_loader=None):
    # 1) model attribute
    if hasattr(model, "num_classes") and isinstance(model.num_classes, int) and model.num_classes > 0:
        return int(model.num_classes)
    # 2) classifier head
    for attr in ("fc", "classifier", "head", "heads", "last_linear"):
        mod = getattr(model, attr, None)
        if mod is not None and hasattr(mod, "out_features"):
            return int(mod.out_features)
    # 3) dataset
    if any_loader is not None:
        ds = any_loader.dataset
        if hasattr(ds, "targets") and len(ds.targets) > 0:
            return int(max(ds.targets)) + 1
        if hasattr(ds, "classes"):
            return int(len(ds.classes))
    raise ValueError("Unable to infer num_classes; pass a loader with targets or a model with an out_features head.")

@torch.inference_mode()
def _top1_and_per_class(model, loader, num_classes, device):
    if loader is None:
        return None, None

    model.eval()
    total = torch.tensor(0, device=device, dtype=torch.long)
    correct = torch.tensor(0, device=device, dtype=torch.long)
    cls_total = torch.zeros(num_classes, device=device, dtype=torch.long)
    cls_correct = torch.zeros(num_classes, device=device, dtype=torch.long)

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        total += targets.numel()
        correct += (preds == targets).sum()

        # per-class
        for c in range(num_classes):
            mask = (targets == c)
            if mask.any():
                cls_total[c] += mask.sum()
                cls_correct[c] += (preds[mask] == c).sum()

    overall = (correct.float() / total.clamp_min(1).float() * 100.0).item()
    per_class = torch.zeros(num_classes, device=device, dtype=torch.float)
    nz = cls_total > 0
    per_class[nz] = (cls_correct[nz].float() / cls_total[nz].float()) * 100.0
    return round(overall, 3), [round(x, 3) for x in per_class.tolist()]

def gather_and_write_metrics_csv(
    csv_path,
    model,
    method,                 # e.g. "retrain" or an unlearning method name
    forget_class,           # int or list; will be stringified
    *,
    # overall retain/forget splits
    train_retain_loader=None,
    train_forget_loader=None,
    test_retain_loader=None,
    test_forget_loader=None,
    # full splits for class-wise tallies (fallback to retain if None)
    train_full_loader=None,
    test_full_loader=None,
    mia_result=None,        # dict/float/str from your evaluation
):
    device = next(model.parameters()).device
    probe_loader = (
        train_full_loader or test_full_loader or
        train_retain_loader or train_forget_loader or
        test_retain_loader or test_forget_loader
    )
    num_classes = _infer_num_classes(model, probe_loader)

    row = {
        "method": str(method),
        "forget_class": json.dumps(forget_class) if not isinstance(forget_class, (str, int)) else forget_class,
    }

    # Overall retain/forget accuracies
    def _overall(loader):
        acc, _ = _top1_and_per_class(model, loader, num_classes, device)
        return acc

    row["train_retain_acc"] = _overall(train_retain_loader)
    row["train_forget_acc"] = _overall(train_forget_loader)
    row["test_retain_acc"]  = _overall(test_retain_loader)
    row["test_forget_acc"]  = _overall(test_forget_loader)

    # Class-wise train/test (on full loaders if given, otherwise retain loaders)
    def _classwise(prefix, loader, fallback):
        use_loader = loader or fallback
        _, per_cls = _top1_and_per_class(model, use_loader, num_classes, device) if use_loader else (None, None)
        # The user asked for classes 0..9 explicitly; fill missing with None if num_classes < 10
        for i in range(10):
            val = None
            if per_cls is not None and i < len(per_cls):
                val = per_cls[i]
            row[f"{prefix}_class{i}_acc"] = val

    _classwise("train", train_full_loader, train_retain_loader)
    _classwise("test",  test_full_loader,  test_retain_loader)

    # MIA (whatever your evaluator returns)
    if mia_result is None:
        row["mia_correctness"] = None
        row["mia_confidence"]  = None
        row["mia_entropy"]     = None
        row["mia_m_entropy"]   = None
        row["mia_prob"]        = None
    elif isinstance(mia_result, dict):
        row["mia_correctness"] = mia_result.get("correctness")
        row["mia_confidence"]  = mia_result.get("confidence")
        row["mia_entropy"]     = mia_result.get("entropy")
        row["mia_m_entropy"]   = mia_result.get("m_entropy")
        row["mia_prob"]        = mia_result.get("prob")
    else:
        row["mia_correctness"] = mia_result
        row["mia_confidence"]  = None
        row["mia_entropy"]     = None
        row["mia_m_entropy"]   = None
        row["mia_prob"]        = None

    # Stable column order
    base_cols   = ["method", "forget_class", "train_retain_acc", "train_forget_acc", "test_retain_acc", "test_forget_acc"]
    class_cols  = [f"train_class{i}_acc" for i in range(10)] + [f"test_class{i}_acc" for i in range(10)]
    mia_cols    = ["mia_correctness", "mia_confidence", "mia_entropy", "mia_m_entropy", "mia_prob"]
    field_order = base_cols + class_cols + mia_cols

    os.makedirs(os.path.dirname(str(csv_path)), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if write_header:
            w.writeheader()
        # ensure all keys exist
        for k in field_order:
            row.setdefault(k, None)
        w.writerow(row)

    return row  # handy for logging/printing

def note_print(*args, **kwargs):

    print("\033[0;32m", *args, "\033[0m", **kwargs) 
    # print("\033[0;40m", *args, "\033[0m", **kwargs) 

def calculate_AUS(A_test_forget, A_test_retain, Aor):
    """
    Args:
        A_test_forget (float): Accuracy on the forget test set.
        A_test_retain (float): Accuracy on the retain test set.
        Aor (float): Accuracy of the original model on the test retain set.
    """
    delta = abs(0 - A_test_forget)
    AUS = (1 - (Aor - A_test_retain)) / (1 + delta)
    return AUS


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Time taken by {func.__name__}: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (hh:mm:ss)")
        return result
    return wrapper

def seed_torch(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class TinyImageNet_load(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)
       
        
        self.targets = [target for _, target in self.images]    

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            # images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]  
            raise ValueError("Python version is too low")
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]  
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):

        # raise NotImplementedError("label if got from samples, however, it should be got from targets (for random label)")
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


class vggface_dataset(torch.utils.data.Dataset):
    def __init__(self, config, transform, mode=None, train=None):    
        if mode == 'pretrain' and train==True:
            self.samples = config["pretrain_train_samples"]
        elif mode == 'pretrain' and train==False:
            self.samples = config["pretrain_test_samples"]
        elif mode == 'finetune' and train==True:
            self.samples = config["finetune_train_samples"]
        elif mode == 'finetune' and train==False:
            self.samples = config["finetune_test_samples"]
        else:
            raise ValueError('mode must be one of [pretrain, finetune]')
        self.targets = [s[1] for s in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # raise NotImplementedError("label if got from samples, however, it should be got from targets (for random label)")
        img_path, _ = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # NOTE: due to random label method, samples.label is the original label, and targets is the noisy label
        # get label from targets
        label = self.targets[index]
        return img, label


class Identity:
    def __call__(self, x):
        return x


def get_transforms(dataset_name, model_name, wo_dataaug): 
    resize_transform = Identity()   if "my" in model_name or model_name == "vgg16" \
                                    else transforms.Resize((224,224))  
    if dataset_name in ["cifar10", "cifar100"]:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            resize_transform,
            transforms.ToTensor(),          
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    elif dataset_name == "vggface": 
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset_name == "tiny_imagenet":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])

        transform_test = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if wo_dataaug:
        transform_train = transform_test
    return transform_train, transform_test


def get_dataset(dataset_name, transform_train, transform_test, path=Path("~/data").expanduser()):
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
    elif dataset_name == 'tiny_imagenet':
        # data_dir = os.path.join(path, 'tiny-imagenet-200') 

        tinyimagenet_dir = os.path.join(path, 'tiny-imagenet-200')
        train_dataset = datasets.ImageFolder(root=os.path.join(tinyimagenet_dir, 'train'), transform=transform_train)
        test_dataset = TinyImageNet_load(tinyimagenet_dir, train=False, transform=transform_test)
    elif dataset_name == 'vggface':
        config_path = 'config/vggface_sample.yaml'
        sample_config = OmegaConf.load(config_path)

        train_dataset = vggface_dataset(sample_config, transform_train, mode='finetune', train=True)
        test_dataset = vggface_dataset(sample_config, transform_test, mode='finetune', train=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, test_dataset


def create_dataset(dir_path, pretrain_class_num=100, finetune_class_num=10):
    label_counts = []
    for subfolder in ["train", "test"]:
        subfolder_path = os.path.join(dir_path, subfolder)
        for class_name in os.listdir(subfolder_path):   
            class_path = os.path.join(subfolder_path, class_name)
            if os.path.isdir(class_path):
                num_images = len(os.listdir(class_path))
                label_counts.append((subfolder+ "/" + class_name, num_images))


    label_counts.sort(key=lambda x: x[1], reverse=True)
    class_num = pretrain_class_num + finetune_class_num

    chosen_classes = [x[0] for x in label_counts if x[1]>= 500]
    assert len(chosen_classes) >= class_num
    chosen_classes = chosen_classes[:class_num]

    print(f"{len(chosen_classes)}，{ 500 }")


    pretrain_train_samples, pretrain_test_samples, finetune_train_samples, finetune_test_samples = [], [], [], []
    for i, class_name in enumerate(chosen_classes):
        class_path = os.path.join(dir_path, class_name)
        assert len(os.listdir(class_path)) >= 500, f"Class {class_name} does not have enough images"

        for j, img_name in enumerate(os.listdir(class_path)): 
            img_path = os.path.join(class_path, img_name)
            if i< pretrain_class_num:
                if j < 100:
                    pretrain_test_samples.append((img_path, i))
                elif j < 500:
                    pretrain_train_samples.append((img_path, i))
                else:
                    break
            else:
                if j < 100:
                    finetune_test_samples.append((img_path, i - pretrain_class_num))   
                elif j < 500:
                    finetune_train_samples.append((img_path, i - pretrain_class_num))
                else:
                    break

    config_dict = {
        "pretrain_train_samples": pretrain_train_samples,
        "pretrain_test_samples": pretrain_test_samples,
        "finetune_train_samples": finetune_train_samples,
        "finetune_test_samples": finetune_test_samples,
    }
    return config_dict


def get_dataloader(trainset, testset, batch_size, num_workers, shuffle=True):
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  

    return train_loader, test_loader


def split_class_data(dataset, forget_class_index, num_forget):

    # targets = torch.tensor([target for _, target in dataset])
    targets = dataset.targets   
    targets = torch.tensor(targets)

    forget_class_index = torch.tensor(forget_class_index)   

    # forget_class_indices = torch.nonzero(targets == forget_class).flatten()
    mask = torch.isin(targets, forget_class_index)
    forget_class_indices = torch.nonzero(mask).flatten()
    

    assert forget_class_indices.numel() > 0, f"No samples found for class in {forget_class_indices}"


    num_forget = min(num_forget, forget_class_indices.numel())


    forget_index = forget_class_indices[:num_forget]


    class_remain_index = forget_class_indices[num_forget:]


    remain_index = torch.nonzero(~mask).flatten()


    remain_index = torch.cat((remain_index, class_remain_index))


    return forget_index.tolist(), remain_index.tolist(), class_remain_index.tolist()


def get_unlearn_loader(trainset, testset, forget_class_index, batch_size, num_forget, num_workers, repair_num_ratio=0.01):

    train_forget_index, train_remain_index, class_remain_index = split_class_data(trainset, forget_class_index, num_forget=num_forget)    # trainset, 4, 5000
    assert isinstance(train_forget_index, list)
    test_forget_index, test_remain_index, _ = split_class_data(testset, forget_class_index, num_forget=len(testset))  # testset, 4, 1000

    ######################################################
    # data_indices={
    #     'train_forget_index': train_forget_index,
    #     'train_remain_index': train_remain_index,
    #     'class_remain_index': class_remain_index,
    #     'test_forget_index': test_forget_index,
    #     'test_remain_index': test_remain_index
    # }
    # # torch.save(data_indices, 'checkpoints/cifar10_data_indices.pth')
    # data_indices_old = torch.load('checkpoints/cifar10_data_indices.pth')
    # for index,index_old in zip(data_indices.values(), data_indices_old.values()):
    #     assert index == index_old
    # print('data_indices check pass!')
    #####################################################

    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)  # 5000
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000

    repair_class_sampler = SubsetRandomSampler(repair_class_index)

    test_forget_sampler = SubsetRandomSampler(test_forget_index)  # 1000
    test_remain_sampler = SubsetRandomSampler(test_remain_index)  # 9000

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_forget_sampler,
                                                      num_workers=num_workers) 
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_remain_sampler,
                                                      num_workers=num_workers)

    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=repair_class_sampler,
                                                      num_workers=num_workers)

    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_forget_sampler,
                                                      num_workers=num_workers)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_remain_sampler,
                                                      num_workers=num_workers)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index


def inf_generator(iterable):

    note_print("inf_generator is deprecated, use itertools.cycle(iterable) instead")
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()   
        except StopIteration:
            iterator = iterable.__iter__()
