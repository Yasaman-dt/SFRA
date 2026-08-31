import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models import load_model
from utils import get_dataset, get_transforms
from project_paths import EXPS_DIR
from relearning_baselines.run_pra_on_single_checkpoint import (
    apply_prototypes_to_classifier,
    collect_prototypes,
    get_active_classifier,
    loader_accuracy,
    parse_alpha_grid,
    parse_forget_arg,
    per_class_accuracy,
    restore_classifier_state,
    save_classifier_state,
)


def _format_class_list(classes):
    return ','.join(str(c) for c in classes)


def _format_per_class(values):
    return ';'.join(
        f'{class_id}:{value:.4f}'
        for class_id, value in values.items()
    )


def checkpoint_for_multi(
    method,
    dataset_name,
    model_name,
    forget_classes,
    lr,
    base_dir,
):
    """Derive the checkpoint path used by the multi-class training code."""
    num_forget = len(set(int(c) for c in forget_classes))
    if num_forget == 0:
        raise ValueError('At least one forget class is required.')

    if method == 'original':
        return os.path.join(
            base_dir,
            'test_pretrained_model',
            f'{dataset_name}_{model_name}_original_model.pth',
        )
    if method == 'retrained':
        return os.path.join(
            base_dir,
            'test_pretrained_model',
            (
                f'{dataset_name}_{model_name}_retrain_'
                f'forget{num_forget}_model.pth'
            ),
        )
    return os.path.join(
        base_dir,
        f'{dataset_name}_{model_name}_forget{num_forget}',
        method,
        f'lr{lr}',
        'ckpt_best_by_aus.pth',
    )


def create_balanced_multi_attack_loaders(
    dataset,
    n_percent,
    forget_classes,
    remain_classes,
    min_forget_per_class,
    batch_size,
    num_workers,
    seed,
):
    """
    Sample the requested percentage, then add forgotten-class examples when
    needed so every forgotten class has enough PRA support images.
    """
    rng = random.Random(seed)
    total = len(dataset)
    sample_size = max(1, int(total * (n_percent / 100.0)))
    sampled_indices = rng.sample(
        range(total),
        min(sample_size, total),
    )
    targets = dataset.targets
    forget_set = set(forget_classes)
    remain_set = set(remain_classes)

    forget_indices = [
        index for index in sampled_indices
        if int(targets[index]) in forget_set
    ]
    remain_indices = [
        index for index in sampled_indices
        if int(targets[index]) in remain_set
    ]

    selected = set(forget_indices)
    per_class_counts = {
        class_id: sum(
            int(targets[index]) == class_id
            for index in forget_indices
        )
        for class_id in forget_classes
    }

    for class_id in forget_classes:
        deficit = min_forget_per_class - per_class_counts[class_id]
        if deficit <= 0:
            continue

        candidates = [
            index for index in range(total)
            if int(targets[index]) == class_id and index not in selected
        ]
        if len(candidates) < deficit:
            raise RuntimeError(
                f'Class {class_id} has only '
                f'{per_class_counts[class_id] + len(candidates)} available '
                f'images, but {min_forget_per_class} are required.'
            )

        added = rng.sample(candidates, deficit)
        forget_indices.extend(added)
        selected.update(added)
        per_class_counts[class_id] += deficit

    generator = torch.Generator()
    generator.manual_seed(seed)
    forget_loader = DataLoader(
        Subset(dataset, forget_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    retain_loader = DataLoader(
        Subset(dataset, remain_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    counts = {
        'unlearn': len(forget_indices),
        'remain': len(remain_indices),
        'unlearn_per_class': per_class_counts,
    }
    print(
        'Attack subset size after support balancing: '
        f'{len(forget_indices) + len(remain_indices)}'
    )
    print(f"  -> Unlearn class samples: {counts['unlearn']}")
    print(f"  -> Remain class samples: {counts['remain']}")
    print(f"  -> Unlearn samples per class: {per_class_counts}")
    return forget_loader, retain_loader, counts


def evaluate_multi_forget_checkpoint(
    checkpoint_path,
    args,
    forget_classes,
    num_classes,
    device,
    attackset,
    testset,
):
    forget_classes = sorted({int(c) for c in forget_classes})
    forget_set = set(forget_classes)
    remain_classes = [
        c for c in range(num_classes)
        if c not in forget_set
    ]

    if not forget_classes:
        raise ValueError('At least one forget class is required.')
    if not remain_classes:
        raise ValueError('At least one retain class is required.')

    print(f'Using multi-forget checkpoint: {checkpoint_path}')
    print(f'Forgotten classes: {_format_class_list(forget_classes)}')
    print(f'Retain classes: {len(remain_classes)} classes')

    model = load_model(
        checkpoint_path,
        args.model_name,
        args.dataset,
        num_classes,
    )
    model = model.to(device).eval()

    forget_attack_loader, retain_attack_loader, counts = (
        create_balanced_multi_attack_loaders(
            dataset=attackset,
            n_percent=args.n_percent,
            forget_classes=forget_classes,
            remain_classes=remain_classes,
            min_forget_per_class=args.num_samples_per_class,
            batch_size=args.attack_batch_size,
            num_workers=args.num_workers,
            seed=args.support_seed,
        )
    )
    print('Attack loader counts:', counts)

    test_loader = DataLoader(
        testset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    baseline_test_accs = per_class_accuracy(
        model,
        test_loader,
        device,
        num_classes,
    )
    baseline_forget_per_class = {
        c: float(baseline_test_accs[c])
        for c in forget_classes
    }
    baseline_retain_per_class = {
        c: float(baseline_test_accs[c])
        for c in remain_classes
    }
    baseline_forget_test = float(np.nanmean(
        list(baseline_forget_per_class.values())
    ))
    baseline_retain_test = float(np.nanmean(
        list(baseline_retain_per_class.values())
    ))

    baseline_retain_selection = loader_accuracy(
        model,
        retain_attack_loader,
        device,
    )

    classifier = get_active_classifier(model)
    original_weight, original_bias = save_classifier_state(classifier)

    prototypes, support_counts = collect_prototypes(
        model=model,
        attack_loader=forget_attack_loader,
        unlearn_classes=forget_classes,
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
            retain_attack_loader,
            device,
        )
        sampled_retain_drop = (
            baseline_retain_selection - retain_after_selection
        )

        # Validate against the same full-test retain metric reported in the
        # final result. The small sampled loader can remain at 100% even when
        # PRA causes a substantial drop across all retained classes.
        candidate_test_accs = per_class_accuracy(
            model,
            test_loader,
            device,
            num_classes,
        )
        candidate_retain_test = float(np.nanmean([
            candidate_test_accs[c] for c in remain_classes
        ]))
        test_retain_drop = baseline_retain_test - candidate_retain_test
        valid = test_retain_drop <= args.max_retain_drop + 1e-12

        candidate_results.append({
            'alpha': float(alpha),
            'retain_selection': float(retain_after_selection),
            'sampled_retain_drop': float(sampled_retain_drop),
            'retain_test': float(candidate_retain_test),
            'retain_drop': float(test_retain_drop),
            'valid': bool(valid),
        })

        status = 'valid' if valid else 'invalid'
        print(
            f'alpha={alpha:.3f}: '
            f'sampled-retain={retain_after_selection:.2f}% '
            f'(drop={sampled_retain_drop:.2f} pp), '
            f'test-retain={candidate_retain_test:.2f}% '
            f'(drop={test_retain_drop:.2f} pp) -> {status}'
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
        selected = max(valid_results, key=lambda result: result['alpha'])
    else:
        selected = candidate_results[0]
        if not selected['valid']:
            raise RuntimeError(
                f'Fixed alpha={selected["alpha"]:.3f} is invalid: '
                f'full test retain accuracy drops by '
                f'{selected["retain_drop"]:.2f} percentage points, '
                f'exceeding --max-retain-drop={args.max_retain_drop:.2f}.'
            )

    selected_alpha = selected['alpha']
    print(
        f'Selected alpha={selected_alpha:.3f} '
        f'with full-test retain drop={selected["retain_drop"]:.2f} pp.'
    )

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

    pra_forget_selection = loader_accuracy(
        model,
        forget_attack_loader,
        device,
    )
    pra_retain_selection = loader_accuracy(
        model,
        retain_attack_loader,
        device,
    )

    attacked_test_accs = per_class_accuracy(
        model,
        test_loader,
        device,
        num_classes,
    )
    pra_forget_per_class = {
        c: float(attacked_test_accs[c])
        for c in forget_classes
    }
    pra_retain_per_class = {
        c: float(attacked_test_accs[c])
        for c in remain_classes
    }
    pra_forget_test = float(np.nanmean(
        list(pra_forget_per_class.values())
    ))
    pra_retain_test = float(np.nanmean(
        list(pra_retain_per_class.values())
    ))

    forget_delta_per_class = {
        c: pra_forget_per_class[c] - baseline_forget_per_class[c]
        for c in forget_classes
    }
    retain_delta_per_class = {
        c: pra_retain_per_class[c] - baseline_retain_per_class[c]
        for c in remain_classes
    }
    forget_delta_test = pra_forget_test - baseline_forget_test
    retain_delta_test = pra_retain_test - baseline_retain_test

    print('\nMulti-class PRA result:')
    print(f'  selected alpha: {selected_alpha:.3f}')
    print(f'  selection forget accuracy: {pra_forget_selection:.2f}%')
    print(f'  selection retain accuracy: {pra_retain_selection:.2f}%')
    print(
        f'  test forget accuracy: {baseline_forget_test:.2f}% '
        f'-> {pra_forget_test:.2f}% '
        f'({forget_delta_test:+.2f} pp)'
    )
    print(
        f'  test retain accuracy: {baseline_retain_test:.2f}% '
        f'-> {pra_retain_test:.2f}% '
        f'({retain_delta_test:+.2f} pp)'
    )

    return {
        'checkpoint': checkpoint_path,
        'forget_classes': forget_classes,
        'forget_class_label': _format_class_list(forget_classes),
        'support_seed': args.support_seed,
        'selected_alpha': selected_alpha,
        'baseline_acc_f_test': baseline_forget_test,
        'baseline_acc_r_test': baseline_retain_test,
        'pra_acc_f_test': pra_forget_test,
        'pra_acc_r_test': pra_retain_test,
        'delta_acc_f_test': forget_delta_test,
        'delta_acc_r_test': retain_delta_test,
        'pra_acc_f_selection': pra_forget_selection,
        'selection_acc_r_before': baseline_retain_selection,
        'selection_acc_r_after': pra_retain_selection,
        'selection_retain_drop': (
            baseline_retain_selection - pra_retain_selection
        ),
        'forget_per_class_before': baseline_forget_per_class,
        'forget_per_class_after': pra_forget_per_class,
        'forget_per_class_delta': forget_delta_per_class,
        'retain_per_class_delta': retain_delta_per_class,
        'counts': counts,
    }


def save_multi_result(result, args):
    out_dir = os.path.join('pra_multi', args.method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f'{args.dataset}_{args.model_name}.csv',
    )

    fieldnames = [
        'forget_classes',
        'support_seed',
        'selected_alpha',
        'baseline_acc_f_test',
        'baseline_acc_r_test',
        'pra_acc_f_test',
        'pra_acc_r_test',
        'delta_acc_f_test',
        'delta_acc_r_test',
        'pra_acc_f_selection',
        'selection_acc_r_before',
        'selection_acc_r_after',
        'selection_retain_drop',
        'unlearn_count',
        'remain_count',
        'forget_per_class_before',
        'forget_per_class_after',
        'forget_per_class_delta',
    ]

    header_needed = not os.path.exists(out_path)
    with open(out_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if header_needed:
            writer.writeheader()
        writer.writerow({
            'forget_classes': result['forget_class_label'],
            'support_seed': result['support_seed'],
            'selected_alpha': f"{result['selected_alpha']:.4f}",
            'baseline_acc_f_test': (
                f"{result['baseline_acc_f_test']:.4f}"
            ),
            'baseline_acc_r_test': (
                f"{result['baseline_acc_r_test']:.4f}"
            ),
            'pra_acc_f_test': f"{result['pra_acc_f_test']:.4f}",
            'pra_acc_r_test': f"{result['pra_acc_r_test']:.4f}",
            'delta_acc_f_test': f"{result['delta_acc_f_test']:.4f}",
            'delta_acc_r_test': f"{result['delta_acc_r_test']:.4f}",
            'pra_acc_f_selection': (
                f"{result['pra_acc_f_selection']:.4f}"
            ),
            'selection_acc_r_before': (
                f"{result['selection_acc_r_before']:.4f}"
            ),
            'selection_acc_r_after': (
                f"{result['selection_acc_r_after']:.4f}"
            ),
            'selection_retain_drop': (
                f"{result['selection_retain_drop']:.4f}"
            ),
            'unlearn_count': result['counts']['unlearn'],
            'remain_count': result['counts']['remain'],
            'forget_per_class_before': _format_per_class(
                result['forget_per_class_before']
            ),
            'forget_per_class_after': _format_per_class(
                result['forget_per_class_after']
            ),
            'forget_per_class_delta': _format_per_class(
                result['forget_per_class_delta']
            ),
        })

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Run PRA on one checkpoint that unlearned several classes '
            'simultaneously.'
        )
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help=(
            'Optional explicit checkpoint path. If omitted, the path is '
            'derived from --base-dir, --method, --dataset, --model, '
            '--forget, and --lr.'
        ),
    )
    parser.add_argument(
        '--base-dir',
        default=str(EXPS_DIR),
        help='Base directory containing the experiment folders.',
    )
    parser.add_argument(
        '--method',
        default='original',
        choices=[
            'original',
            'retrained',
            'random_label',
            'finetune',
            'gradient_ascent',
            'neggrad_plus',
            'boundary_shrink',
            'boundary_expand',
            'l2ul_adv',
            'l2ul_imp',
            'fisher',
            'wood_fisher',
            'delete',
            'bad_teacher',
            'salun',
            'scrub',
        ],
        help='Unlearning method used to derive the checkpoint path.',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate used in the unlearning checkpoint path.',
    )
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['cifar10', 'cifar100', 'tiny_imagenet'],
    )
    parser.add_argument('--model', dest='model_name', default='resnet18')
    parser.add_argument(
        '--forget',
        required=True,
        help='Comma/range list of classes forgotten together, e.g. 0,1,2 or 0-4.',
    )
    parser.add_argument(
        '--n-percent',
        type=float,
        default=1.0,
        help='Percent of attack split sampled for PRA support/selection.',
    )
    parser.add_argument(
        '--num-samples-per-class',
        type=int,
        default=5,
        help='Number of real support samples per forgotten class.',
    )
    parser.add_argument(
        '--pra-alpha',
        type=float,
        default=None,
        help='Use one fixed alpha. If omitted, alpha is selected automatically.',
    )
    parser.add_argument(
        '--alpha-grid',
        type=parse_alpha_grid,
        default=parse_alpha_grid(
            '1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0'
        ),
        help='Comma-separated alpha candidates for automatic selection.',
    )
    parser.add_argument(
        '--max-retain-drop',
        type=float,
        default=1.0,
        help='Maximum allowed retain-accuracy drop in percentage points.',
    )
    parser.add_argument(
        '--support-seed',
        type=int,
        default=0,
        help='Seed controlling selection of PRA support images.',
    )
    parser.add_argument(
        '--pra-metric',
        choices=['cosine', 'l2'],
        default='cosine',
    )
    parser.add_argument(
        '--no-normalize-proto',
        action='store_true',
        help='Do not L2-normalize class prototypes.',
    )
    parser.add_argument(
        '--attack-split',
        choices=['train', 'test'],
        default='train',
        help='Dataset split used to obtain real PRA support samples.',
    )
    parser.add_argument('--attack-batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    num_classes_by_dataset = {
        'cifar10': 10,
        'cifar100': 100,
        'tiny_imagenet': 200,
    }
    num_classes = num_classes_by_dataset[args.dataset]
    forget_classes = parse_forget_arg(args.forget, num_classes)

    if len(forget_classes) < 2:
        print(
            'WARNING: this multi script is intended for two or more '
            'simultaneously forgotten classes.'
        )

    if args.checkpoint is not None:
        checkpoint_path = os.path.expanduser(args.checkpoint)
    else:
        checkpoint_path = checkpoint_for_multi(
            method=args.method,
            dataset_name=args.dataset,
            model_name=args.model_name,
            forget_classes=forget_classes,
            lr=args.lr,
            base_dir=os.path.expanduser(args.base_dir),
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'Checkpoint not found: '
            f'{checkpoint_path}\n'
            'Pass --checkpoint explicitly if the model was saved under '
            'a different naming convention.'
        )

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    transform_train, transform_test = get_transforms(
        args.dataset,
        args.model_name,
        wo_dataaug=True,
    )
    trainset, testset = get_dataset(
        args.dataset,
        transform_train,
        transform_test,
    )
    attackset = trainset if args.attack_split == 'train' else testset
    print(f'PRA support samples are drawn from the {args.attack_split} split.')

    result = evaluate_multi_forget_checkpoint(
        checkpoint_path=checkpoint_path,
        args=args,
        forget_classes=forget_classes,
        num_classes=num_classes,
        device=device,
        attackset=attackset,
        testset=testset,
    )

    print('\nSummary')
    print(
        f"forget={result['forget_class_label']} "
        f"alpha={result['selected_alpha']:.3f} "
        f"before_f={result['baseline_acc_f_test']:.2f}% "
        f"before_r={result['baseline_acc_r_test']:.2f}% "
        f"pra_f={result['pra_acc_f_test']:.2f}% "
        f"pra_r={result['pra_acc_r_test']:.2f}% "
        f"delta_f={result['delta_acc_f_test']:+.2f}pp "
        f"delta_r={result['delta_acc_r_test']:+.2f}pp"
    )

    out_path = save_multi_result(result, args)
    print(f'CSV saved to: {out_path}')


if __name__ == '__main__':
    main()
