import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Subset

from models import load_model
from utils import get_transforms, get_dataset


def _last_linear(module):
    """Return the last nn.Linear contained in a module."""
    if isinstance(module, nn.Linear):
        return module
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return linear_layers[-1] if linear_layers else None


def get_active_classifier(model):
    """
    Locate the linear layer that is actually used to produce logits.

    Order matters: the custom Swin model contains both ``head.fc`` and
    ``fc``. Its ClassifierHead bypasses ``head.fc`` and ``forward()`` sends
    the pre-logit embedding through the top-level ``fc`` instead.
    """
    base = model.module if hasattr(model, "module") else model

    # Custom Swin used in this repository: model.head returns pre-logits and
    # model.forward() applies this top-level layer to produce the logits.
    if (
        hasattr(base, "head")
        and hasattr(base.head, "fc")
        and hasattr(base, "fc")
        and isinstance(base.fc, nn.Linear)
    ):
        return base.fc

    # Standard timm-style Swin classifiers use model.head.fc directly.
    if hasattr(base, "head") and hasattr(base.head, "fc"):
        if isinstance(base.head.fc, nn.Linear):
            return base.head.fc

    # ViT_16_mod and torchvision-style ViTs.
    if hasattr(base, "heads"):
        layer = _last_linear(base.heads)
        if layer is not None:
            return layer

    # ResNet, VGG, custom VisionTransformer, or Sequential ResNet head.
    if hasattr(base, "fc"):
        layer = _last_linear(base.fc)
        if layer is not None:
            return layer

    # Generic fallback.
    if hasattr(base, "classifier"):
        layer = _last_linear(base.classifier)
        if layer is not None:
            return layer

    raise AttributeError(
        f"Could not locate the active classifier for {type(base).__name__}."
    )


def extract_features(model, images, classifier=None):
    """
    Capture the exact tensor entering the active final linear classifier.

    A forward pre-hook makes this architecture-independent and guarantees
    that the prototype dimension matches classifier.in_features.
    """
    classifier = classifier or get_active_classifier(model)
    captured = {}

    def _capture_input(_module, inputs):
        if not inputs:
            raise RuntimeError("The classifier pre-hook received no input.")
        captured["features"] = inputs[0].detach()

    handle = classifier.register_forward_pre_hook(_capture_input)
    try:
        _ = model(images)
    finally:
        handle.remove()

    if "features" not in captured:
        raise RuntimeError(
            f"Failed to capture features for {type(model).__name__}."
        )

    feats = captured["features"]
    if feats.ndim != 2:
        feats = torch.flatten(feats, 1)

    if feats.shape[1] != classifier.in_features:
        raise RuntimeError(
            "Feature/classifier mismatch after extraction: "
            f"features={tuple(feats.shape)}, "
            f"classifier.in_features={classifier.in_features}."
        )
    return feats


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

    classifier = get_active_classifier(model)
    print(
        f"Active classifier: {classifier} "
        f"(in_features={classifier.in_features}, "
        f"out_features={classifier.out_features})"
    )

    feature_dict = {c: [] for c in unlearn_classes}
    counts = {c: 0 for c in unlearn_classes}

    with torch.no_grad():
        for images, labels in attack_loader:
            images, labels = images.to(device), labels.to(device)
            feats = extract_features(model, images, classifier)

            for feat, lbl in zip(feats, labels):
                c = int(lbl.item())
                if c in counts and counts[c] < num_samples_per_class:
                    feature_dict[c].append(feat.clone())
                    counts[c] += 1

            if all(counts[c] >= num_samples_per_class for c in unlearn_classes):
                break

    missing = [
        c for c in unlearn_classes
        if len(feature_dict[c]) < num_samples_per_class
    ]
    if missing:
        raise RuntimeError(
            "Not enough attack samples for classes "
            f"{missing}. Collected counts: {counts}"
        )

    proto_dict = {}
    for c, flist in feature_dict.items():
        proto = torch.stack(flist, dim=0).mean(dim=0)
        if normalize_proto:
            proto = F.normalize(proto, p=2, dim=0)
        proto_dict[c] = proto.to(device)

    orig_w = classifier.weight.detach().clone()
    if classifier.bias is not None:
        orig_b = classifier.bias.detach().clone()
    else:
        orig_b = torch.zeros(
            classifier.out_features,
            device=device,
            dtype=orig_w.dtype,
        )

    with torch.no_grad():
        for c, proto in proto_dict.items():
            if proto.numel() != classifier.in_features:
                raise RuntimeError(
                    f"Class {c}: prototype dimension {proto.numel()} does "
                    f"not match classifier dimension {classifier.in_features}."
                )

            if metric == 'l2':
                w_proto = 2.0 * proto
                b_proto = -proto.pow(2).sum()
            elif metric == 'cosine':
                w_proto = proto
                b_proto = torch.tensor(
                    0.0, device=device, dtype=orig_w.dtype
                )
            else:
                raise ValueError(f"Unsupported PRA metric: {metric}")

            w_new = alpha * w_proto + (1.0 - alpha) * orig_w[c]
            b_new = alpha * b_proto + (1.0 - alpha) * orig_b[c]

            classifier.weight[c].copy_(w_new)
            if classifier.bias is not None:
                classifier.bias[c].copy_(b_new)

    return model



def collect_prototypes(
    model,
    attack_loader,
    unlearn_classes,
    num_samples_per_class=5,
    normalize_proto=True,
    device='cuda',
):
    """
    Extract exactly k real support examples per forget class and compute
    one pre-head prototype per class. The same prototypes are reused for
    every alpha candidate, ensuring a fair alpha comparison.
    """
    model = model.to(device).eval()
    classifier = get_active_classifier(model)

    feature_dict = {c: [] for c in unlearn_classes}
    counts = {c: 0 for c in unlearn_classes}

    with torch.no_grad():
        for images, labels in attack_loader:
            images = images.to(device)
            labels = labels.to(device)
            feats = extract_features(model, images, classifier)

            for feat, label in zip(feats, labels):
                c = int(label.item())
                if c in counts and counts[c] < num_samples_per_class:
                    feature_dict[c].append(feat.detach().clone())
                    counts[c] += 1

            if all(counts[c] >= num_samples_per_class for c in unlearn_classes):
                break

    missing = [
        c for c in unlearn_classes
        if counts[c] < num_samples_per_class
    ]
    if missing:
        raise RuntimeError(
            f"Could not collect {num_samples_per_class} support samples "
            f"for classes {missing}. Collected counts: {counts}"
        )

    prototypes = {}
    for c, features in feature_dict.items():
        proto = torch.stack(features, dim=0).mean(dim=0)
        if normalize_proto:
            proto = F.normalize(proto, p=2, dim=0)
        prototypes[c] = proto.to(device)

    return prototypes, counts


def save_classifier_state(classifier):
    weight = classifier.weight.detach().clone()
    bias = (
        classifier.bias.detach().clone()
        if classifier.bias is not None
        else None
    )
    return weight, bias


def restore_classifier_state(classifier, weight, bias):
    with torch.no_grad():
        classifier.weight.copy_(weight)
        if classifier.bias is not None and bias is not None:
            classifier.bias.copy_(bias)


def apply_prototypes_to_classifier(
    classifier,
    prototypes,
    original_weight,
    original_bias,
    alpha,
    metric='cosine',
):
    """Apply PRA to forget-class rows only."""
    with torch.no_grad():
        for c, proto in prototypes.items():
            if proto.numel() != classifier.in_features:
                raise RuntimeError(
                    f"Class {c}: prototype dimension {proto.numel()} does "
                    f"not match classifier dimension {classifier.in_features}."
                )

            if metric == 'cosine':
                w_proto = proto
                b_proto = torch.tensor(
                    0.0,
                    device=original_weight.device,
                    dtype=original_weight.dtype,
                )
            elif metric == 'l2':
                w_proto = 2.0 * proto
                b_proto = -proto.pow(2).sum()
            else:
                raise ValueError(f"Unsupported PRA metric: {metric}")

            classifier.weight[c].copy_(
                alpha * w_proto + (1.0 - alpha) * original_weight[c]
            )
            if classifier.bias is not None:
                old_bias = original_bias[c]
                classifier.bias[c].copy_(
                    alpha * b_proto + (1.0 - alpha) * old_bias
                )


def loader_accuracy(model, loader, device):
    """Overall percentage accuracy on one loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return 100.0 * correct / total if total > 0 else float('nan')


def parse_alpha_grid(value):
    values = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        alpha = float(item)
        if not 0.0 <= alpha <= 1.0:
            raise argparse.ArgumentTypeError(
                f"Alpha must be in [0, 1], received {alpha}."
            )
        values.append(alpha)

    if not values:
        raise argparse.ArgumentTypeError("Alpha grid cannot be empty.")

    # Highest alpha first; remove duplicates.
    return sorted(set(values), reverse=True)


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

    generator = torch.Generator()
    generator.manual_seed(0 if seed is None else seed)

    unlearn_attack_loader = DataLoader(
        unlearn_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )
    remain_attack_loader = DataLoader(
        remain_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )

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


def evaluate_one_checkpoint(
    checkpoint_path,
    args,
    forget_class,
    num_classes,
    device,
    attackset,
    testset,
    transform_train,
    transform_test,
):
    remain_classes = [
        c for c in range(num_classes)
        if c != forget_class
    ]

    print(f'Using checkpoint: {checkpoint_path}')
    model = load_model(
        checkpoint_path,
        args.model_name,
        args.dataset,
        num_classes,
    )
    model = model.to(device).eval()

    _, unlearn_attack_loader, remain_attack_loader, counts = (
        create_attack_loaders(
            test_set=attackset,
            n_percent=args.n_percent,
            unlearn_classes=[forget_class],
            remain_classes=remain_classes,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            seed=args.support_seed,
        )
    )
    print('Attack loader counts:', counts)

    test_loader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
    )

    # Unlearned-model test accuracies before PRA.
    baseline_test_accs = per_class_accuracy(
        model,
        test_loader,
        device,
        num_classes,
    )
    baseline_forget_test = float(
        baseline_test_accs[forget_class]
    )
    baseline_retain_test = float(np.nanmean([
        baseline_test_accs[c] for c in remain_classes
    ]))

    # The paper constrains the post-PRA retain accuracy relative to the
    # retain accuracy before PRA. We evaluate this constraint on the
    # source-dependent attack split, not on the final test split.
    baseline_retain_selection = loader_accuracy(
        model,
        remain_attack_loader,
        device,
    )

    classifier = get_active_classifier(model)
    original_weight, original_bias = save_classifier_state(classifier)

    prototypes, support_counts = collect_prototypes(
        model=model,
        attack_loader=unlearn_attack_loader,
        unlearn_classes=[forget_class],
        num_samples_per_class=args.num_samples_per_class,
        normalize_proto=not args.no_normalize_proto,
        device=device,
    )
    print('Prototype support counts:', support_counts)

    if args.pra_alpha is not None:
        alpha_candidates = [args.pra_alpha]
        automatic_selection = False
    else:
        alpha_candidates = args.alpha_grid
        automatic_selection = True

    candidate_results = []
    for alpha in alpha_candidates:
        restore_classifier_state(
            classifier,
            original_weight,
            original_bias,
        )
        apply_prototypes_to_classifier(
            classifier=classifier,
            prototypes=prototypes,
            original_weight=original_weight,
            original_bias=original_bias,
            alpha=alpha,
            metric=args.pra_metric,
        )

        retain_after_selection = loader_accuracy(
            model,
            remain_attack_loader,
            device,
        )
        retain_drop = (
            baseline_retain_selection - retain_after_selection
        )
        valid = retain_drop <= args.max_retain_drop + 1e-12

        candidate_results.append({
            'alpha': float(alpha),
            'retain_selection': float(retain_after_selection),
            'retain_drop': float(retain_drop),
            'valid': bool(valid),
        })

        status = 'valid' if valid else 'invalid'
        print(
            f'alpha={alpha:.3f}: selection-retain='
            f'{retain_after_selection:.2f}% '
            f'(drop={retain_drop:.2f} pp) -> {status}'
        )

    valid_results = [
        result for result in candidate_results
        if result['valid']
    ]

    if automatic_selection:
        if not valid_results:
            raise RuntimeError(
                'No alpha satisfies the retain-accuracy constraint. '
                'Include alpha=0 in --alpha-grid or increase '
                '--max-retain-drop.'
            )

        # Choose the strongest interpolation that satisfies the paper's
        # retain-accuracy constraint.
        selected = max(
            valid_results,
            key=lambda result: result['alpha'],
        )
    else:
        selected = candidate_results[0]
        if not selected['valid']:
            print(
                'WARNING: the fixed alpha violates the requested '
                'retain-accuracy-drop constraint.'
            )

    selected_alpha = selected['alpha']
    print(
        f'Selected alpha={selected_alpha:.3f} '
        f'with retain drop={selected["retain_drop"]:.2f} pp.'
    )

    # Restore the clean unlearned head, then apply the selected PRA once.
    restore_classifier_state(
        classifier,
        original_weight,
        original_bias,
    )
    apply_prototypes_to_classifier(
        classifier=classifier,
        prototypes=prototypes,
        original_weight=original_weight,
        original_bias=original_bias,
        alpha=selected_alpha,
        metric=args.pra_metric,
    )

    # Training/attack-split metrics correspond to Proto-Acc_f and Acc_r*
    # in the PRA paper. Test metrics are suitable for your WACV table.
    pra_forget_selection = loader_accuracy(
        model,
        unlearn_attack_loader,
        device,
    )
    pra_retain_selection = loader_accuracy(
        model,
        remain_attack_loader,
        device,
    )

    attacked_test_accs = per_class_accuracy(
        model,
        test_loader,
        device,
        num_classes,
    )
    pra_forget_test = float(attacked_test_accs[forget_class])
    pra_retain_test = float(np.nanmean([
        attacked_test_accs[c] for c in remain_classes
    ]))

    print('\nPRA result:')
    print(
        f'  selected alpha: {selected_alpha:.3f}'
    )
    print(
        f'  selection forget accuracy: '
        f'{pra_forget_selection:.2f}%'
    )
    print(
        f'  selection retain accuracy: '
        f'{pra_retain_selection:.2f}%'
    )
    print(
        f'  test forget accuracy: {pra_forget_test:.2f}%'
    )
    print(
        f'  test retain accuracy: {pra_retain_test:.2f}%'
    )

    return {
        'forget_class': forget_class,
        'checkpoint': checkpoint_path,
        'support_seed': args.support_seed,
        'selected_alpha': selected_alpha,
        'baseline_acc_f_test': baseline_forget_test,
        'baseline_acc_r_test': baseline_retain_test,
        'acc_f': pra_forget_test,
        'acc_r': pra_retain_test,
        'acc_f_selection': pra_forget_selection,
        'acc_r_selection_before': baseline_retain_selection,
        'acc_r_selection_after': pra_retain_selection,
        'retain_drop_selection': (
            baseline_retain_selection - pra_retain_selection
        ),
        'counts': counts,
    }


def append_results_to_csv(results, args):
    out_dir = os.path.join('pra', args.method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f'{args.dataset}_{args.model_name}.csv',
    )

    import csv

    fieldnames = [
        'forget_class',
        'support_seed',
        'selected_alpha',
        'baseline_acc_f_test',
        'baseline_acc_r_test',
        'pra_acc_f_test',
        'pra_acc_r_test',
        'pra_acc_f_selection',
        'selection_acc_r_before',
        'selection_acc_r_after',
        'selection_retain_drop',
        'unlearn_count',
        'remain_count',
    ]

    header_needed = not os.path.exists(out_path)
    with open(out_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if header_needed:
            writer.writeheader()

        for item in results:
            writer.writerow({
                'forget_class': item['forget_class'],
                'support_seed': item['support_seed'],
                'selected_alpha': (
                    f"{item['selected_alpha']:.4f}"
                ),
                'baseline_acc_f_test': (
                    f"{item['baseline_acc_f_test']:.4f}"
                ),
                'baseline_acc_r_test': (
                    f"{item['baseline_acc_r_test']:.4f}"
                ),
                'pra_acc_f_test': f"{item['acc_f']:.4f}",
                'pra_acc_r_test': f"{item['acc_r']:.4f}",
                'pra_acc_f_selection': (
                    f"{item['acc_f_selection']:.4f}"
                ),
                'selection_acc_r_before': (
                    f"{item['acc_r_selection_before']:.4f}"
                ),
                'selection_acc_r_after': (
                    f"{item['acc_r_selection_after']:.4f}"
                ),
                'selection_retain_drop': (
                    f"{item['retain_drop_selection']:.4f}"
                ),
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
    parser.add_argument(
        '--pra-alpha',
        type=float,
        default=None,
        help=(
            'Use one fixed alpha. By default alpha is selected '
            'automatically under the retain-drop constraint.'
        ),
    )
    parser.add_argument(
        '--alpha-grid',
        type=parse_alpha_grid,
        default=parse_alpha_grid(
            '1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0'
        ),
        help=(
            'Comma-separated alpha candidates for automatic selection. '
            'The paper reports an ablation at 1.0,0.8,0.6,0.4,0.2, '
            'but does not publish every Table-1 alpha.'
        ),
    )
    parser.add_argument(
        '--max-retain-drop',
        type=float,
        default=1.0,
        help=(
            'Maximum allowed retain-accuracy drop in percentage '
            'points during alpha selection.'
        ),
    )
    parser.add_argument(
        '--support-seed',
        type=int,
        default=0,
        help='Seed controlling selection of the real PRA support images.',
    )
    parser.add_argument('--pra-metric', choices=['cosine', 'l2'], default='cosine',
                        help='Prototype-to-classifier conversion used by PRA')
    parser.add_argument('--no-normalize-proto', action='store_true',
                        help='Do not L2-normalize the class prototype')
    parser.add_argument('--attack-split', choices=['train', 'test'], default='train',
                        help='Dataset split used to obtain the real PRA support samples. Train avoids test leakage.')
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
    attackset = trainset if args.attack_split == 'train' else testset
    print(f'PRA support samples are drawn from the {args.attack_split} split.')

    results = []
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
        if len(forget_classes) != 1:
            print('A single explicit checkpoint was provided, so only the first forget class will be evaluated.')
        results.append(evaluate_one_checkpoint(args.checkpoint, args, forget_classes[0], num_classes, device, attackset, testset, transform_train, transform_test))
    else:
        for forget_class in forget_classes:
            checkpoint_path = checkpoint_for(args.method, args.dataset, args.model_name, forget_class, args.lr, args.base_dir)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
            results.append(evaluate_one_checkpoint(checkpoint_path, args, forget_class, num_classes, device, attackset, testset, transform_train, transform_test))

    print('\nSummary')
    for item in results:
        print(
            f"forget={item['forget_class']} "
            f"alpha={item['selected_alpha']:.3f} "
            f"before_f={item['baseline_acc_f_test']:.2f}% "
            f"before_r={item['baseline_acc_r_test']:.2f}% "
            f"pra_f={item['acc_f']:.2f}% "
            f"pra_r={item['acc_r']:.2f}% "
            f"checkpoint={item['checkpoint']}"
        )

    mean_acc_f = float(np.nanmean([item['acc_f'] for item in results]))
    mean_acc_r = float(np.nanmean([item['acc_r'] for item in results]))
    print(f'\nMean over runs: acc_f={mean_acc_f:.2f}% acc_r={mean_acc_r:.2f}%')

    out_path = append_results_to_csv(results, args)
    print(f'CSV saved to: {out_path}')


if __name__ == '__main__':
    main()
