import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import torch.nn.functional as F
from torch.utils.data import Subset

from models import load_model
from utils import get_transforms, get_dataset


def extract_features(model, images):
    if hasattr(model, 'features') and hasattr(model, 'avg_pool'):
        feats = model.avg_pool(model.features(images))
        return feats.view(feats.size(0), -1)

    if hasattr(model, 'features') and hasattr(model, 'avgpool'):
        feats = model.avgpool(model.features(images))
        return feats.view(feats.size(0), -1)

    if hasattr(model, 'avgpool') and hasattr(model, 'layer4'):
        x = model.conv1(images)
        if hasattr(model, 'bn1'):
            x = model.bn1(x)
        if hasattr(model, 'relu'):
            x = model.relu(x)
        if hasattr(model, 'maxpool'):
            x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    raise AttributeError(f'Unsupported model architecture for feature extraction: {type(model).__name__}')


def update_fc_with_prototypes(
    model,
    attack_loader,
    unlearn_classes,
    num_samples_per_class=5,
    metric='cosine',
    normalize_proto=True,
    alpha=0.5,
    device='cuda',
):
    model = model.to(device)
    model.eval()

    feature_dict = {c: [] for c in unlearn_classes}
    counts = {c: 0 for c in unlearn_classes}

    with torch.no_grad():
        for images, labels in attack_loader:
            images, labels = images.to(device), labels.to(device)
            feats = extract_features(model, images)

            for feat, lbl in zip(feats, labels):
                c = int(lbl.item())
                if c in counts and counts[c] < num_samples_per_class:
                    feature_dict[c].append(feat.clone())
                    counts[c] += 1

            if all(counts[c] >= num_samples_per_class for c in unlearn_classes):
                break

    proto_dict = {}
    for c, flist in feature_dict.items():
        if len(flist) == 0:
            continue
        proto = torch.stack(flist, dim=0).mean(dim=0)
        if normalize_proto:
            proto = F.normalize(proto, p=2, dim=0)
        proto_dict[c] = proto.to(device)

    orig_w = model.fc.weight.data.clone()
    orig_b = model.fc.bias.data.clone() if model.fc.bias is not None else torch.zeros(orig_w.size(0), device=device)

    for c, proto in proto_dict.items():
        if metric == 'l2':
            w_proto = 2.0 * proto
            b_proto = -proto.pow(2).sum()
        else:
            w_proto = proto
            b_proto = torch.tensor(0.0, device=device)

        w_new = alpha * w_proto + (1 - alpha) * orig_w[c]
        b_new = alpha * b_proto + (1 - alpha) * orig_b[c]
        model.fc.weight.data[c] = w_new
        model.fc.bias.data[c] = b_new

    return model


def create_attack_loaders(
    test_set,
    n_percent,
    unlearn_classes,
    remain_classes,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    seed=None,
):
    total = len(test_set)
    k = max(1, int(total * (n_percent / 100.0)))
    if seed is not None:
        random.seed(seed)
    all_indices = list(range(total))
    sampled_indices = random.sample(all_indices, min(k, total))
    targets = test_set.targets
    unlearn_idx = [idx for idx in sampled_indices if targets[idx] in unlearn_classes]
    remain_idx = [idx for idx in sampled_indices if targets[idx] in remain_classes]
    unlearn_subset = Subset(test_set, unlearn_idx)
    remain_subset = Subset(test_set, remain_idx)

    unlearn_attack_loader = DataLoader(unlearn_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    remain_attack_loader = DataLoader(remain_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    counts = {'unlearn': len(unlearn_idx), 'remain': len(remain_idx)}
    print(f"Attack subset size: {len(sampled_indices)}")
    print(f"  -> Unlearn class samples: {counts['unlearn']}")
    print(f"  -> Remain class samples: {counts['remain']}")
    return test_set, unlearn_attack_loader, remain_attack_loader, counts


def parse_forget_arg(s, num_classes):
    s = (s or 'all').strip().lower()
    if s in ('all', '-1'):
        return list(range(num_classes))
    selected = set()
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            for c in range(a, b + 1):
                if 0 <= c < num_classes:
                    selected.add(c)
        else:
            c = int(part)
            if 0 <= c < num_classes:
                selected.add(c)
    return sorted(selected)


def checkpoint_for(method, dataset_name, model_name, forget_class, lr, base_dir):
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    if method == 'retrained':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"
    return f"{base_dir}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"


def per_class_accuracy(model, loader, device, num_classes):
    model.eval()
    correct = np.zeros(num_classes, dtype=int)
    total = np.zeros(num_classes, dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                total[t] += 1
                if p == t:
                    correct[t] += 1
    acc = {c: (100.0 * correct[c] / total[c] if total[c] > 0 else float('nan')) for c in range(num_classes)}
    return acc


def evaluate_one_checkpoint(checkpoint_path, args, forget_class, num_classes, device, testset, transform_train, transform_test):
    remain_classes = [c for c in range(num_classes) if c != forget_class]

    print(f'Using checkpoint: {checkpoint_path}')
    model = load_model(checkpoint_path, args.model_name, args.dataset, num_classes)
    model = model.to(device).eval()

    attack_subset, unlearn_attack_loader, remain_attack_loader, counts = create_attack_loaders(
        test_set=testset,
        n_percent=args.n_percent,
        unlearn_classes=[forget_class],
        remain_classes=remain_classes,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        seed=0,
    )

    print('Attack loader counts:', counts)

    model_attacked = update_fc_with_prototypes(
        model=model,
        attack_loader=unlearn_attack_loader,
        unlearn_classes=[forget_class],
        num_samples_per_class=args.num_samples_per_class,
        metric='cosine',
        normalize_proto=True,
        alpha=1.0,
        device=device,
    )

    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    accs = per_class_accuracy(model_attacked, test_loader, device, num_classes)
    forget_acc = float(accs[forget_class])
    retain_acc = float(np.nanmean([accs[c] for c in remain_classes])) if remain_classes else float('nan')

    print('\nPer-class test accuracy (attacked model):')
    print(f'  forget class {forget_class}: {forget_acc:.2f}%')
    print(f'  mean retain accuracy: {retain_acc:.2f}%')

    return {
        'forget_class': forget_class,
        'checkpoint': checkpoint_path,
        'acc_f': forget_acc,
        'acc_r': retain_acc,
        'counts': counts,
    }


def append_results_to_csv(results, args):
    out_dir = os.path.join('pra', args.method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.dataset}_{args.model_name}.csv')

    import csv

    fieldnames = ['forget_class', 'acc_f', 'acc_r', 'unlearn_count', 'remain_count']

    header_needed = not os.path.exists(out_path)
    with open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if header_needed:
            writer.writeheader()
        for item in results:
            writer.writerow({
                'forget_class': item['forget_class'],
                'acc_f': f"{item['acc_f']:.4f}",
                'acc_r': f"{item['acc_r']:.4f}",
                'unlearn_count': item['counts']['unlearn'],
                'remain_count': item['counts']['remain'],
            })

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None,
                        help='Optional explicit checkpoint path (.pth). If omitted, it is derived from the other arguments.')
    parser.add_argument('--base-dir', default='/export/livia/home/vision/Zdehghani/classification/exps',
                        help='Base directory that contains the experiment folders')
    parser.add_argument('--method', default='original',
                        choices=['original', 'retrained', 'random_label', 'finetune', 'gradient_ascent', 'neggrad_plus',
                                 'boundary_shrink', 'boundary_expand', 'l2ul_adv', 'l2ul_imp', 'fisher', 'wood_fisher',
                                 'delete', 'bad_teacher', 'salun', 'scrub'],
                        help='Unlearning method folder name used to derive the checkpoint path')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate used in the unlearning run; needed to derive the checkpoint path')
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'tiny_imagenet', 'imagenet'])
    parser.add_argument('--model', dest='model_name', default='resnet18')
    parser.add_argument('--forget', default='0')
    parser.add_argument('--n-percent', type=float, default=1.0,
                        help='Percent of test set to sample for attack loader (e.g., 1.0)')
    parser.add_argument('--num-samples-per-class', type=int, default=5,
                        help='Number of real samples per unlearn class to build prototype')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}
    num_classes = NUM_CLASSES[args.dataset]

    forget_classes = parse_forget_arg(args.forget, num_classes)
    remain_classes = [c for c in range(num_classes) if c not in forget_classes]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # dataset
    transform_train, transform_test = get_transforms(args.dataset, args.model_name, wo_dataaug=True)
    trainset, testset = get_dataset(args.dataset, transform_train, transform_test)

    results = []
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
        if len(forget_classes) != 1:
            print('A single explicit checkpoint was provided, so only the first forget class will be evaluated.')
        results.append(evaluate_one_checkpoint(args.checkpoint, args, forget_classes[0], num_classes, device, testset, transform_train, transform_test))
    else:
        for forget_class in forget_classes:
            checkpoint_path = checkpoint_for(args.method, args.dataset, args.model_name, forget_class, args.lr, args.base_dir)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
            results.append(evaluate_one_checkpoint(checkpoint_path, args, forget_class, num_classes, device, testset, transform_train, transform_test))

    print('\nSummary')
    for item in results:
        print(f"forget={item['forget_class']} acc_f={item['acc_f']:.2f}% acc_r={item['acc_r']:.2f}% checkpoint={item['checkpoint']}")

    mean_acc_f = float(np.nanmean([item['acc_f'] for item in results]))
    mean_acc_r = float(np.nanmean([item['acc_r'] for item in results]))
    print(f'\nMean over runs: acc_f={mean_acc_f:.2f}% acc_r={mean_acc_r:.2f}%')

    out_path = append_results_to_csv(results, args)
    print(f'CSV saved to: {out_path}')


if __name__ == '__main__':
    main()
