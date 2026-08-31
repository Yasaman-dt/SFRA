import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import ViT_16_mod, get_model
from utils import get_dataset, get_transforms
from project_paths import EXPS_DIR


NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
}


def _last_linear(module):
    if isinstance(module, nn.Linear):
        return module
    linear_layers = [
        layer for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]
    return linear_layers[-1] if linear_layers else None


def get_active_classifier(model):
    """Locate the final linear layer actually used to produce logits."""
    base = model.module if hasattr(model, 'module') else model

    # This repository's custom Swin bypasses head.fc and uses top-level fc.
    if (
        hasattr(base, 'head')
        and hasattr(base.head, 'fc')
        and hasattr(base, 'fc')
        and isinstance(base.fc, nn.Linear)
    ):
        return base.fc

    if hasattr(base, 'head') and hasattr(base.head, 'fc'):
        if isinstance(base.head.fc, nn.Linear):
            return base.head.fc

    if hasattr(base, 'heads'):
        layer = _last_linear(base.heads)
        if layer is not None:
            return layer

    if hasattr(base, 'fc'):
        layer = _last_linear(base.fc)
        if layer is not None:
            return layer

    if hasattr(base, 'classifier'):
        layer = _last_linear(base.classifier)
        if layer is not None:
            return layer

    raise AttributeError(
        f'Could not locate the active classifier for '
        f'{type(base).__name__}.'
    )


def extract_features(model, images, classifier=None):
    """Capture the exact tensor entering the active final classifier."""
    classifier = classifier or get_active_classifier(model)
    captured = {}

    def capture_input(_module, inputs):
        if inputs:
            captured['features'] = inputs[0].detach()

    handle = classifier.register_forward_pre_hook(capture_input)
    try:
        model(images)
    finally:
        handle.remove()

    if 'features' not in captured:
        raise RuntimeError('Failed to capture classifier input features.')
    features = captured['features']
    if features.ndim != 2:
        features = torch.flatten(features, 1)
    if features.shape[1] != classifier.in_features:
        raise RuntimeError(
            f'Feature dimension {features.shape[1]} does not match '
            f'classifier dimension {classifier.in_features}.'
        )
    return features


def parse_forget_arg(value, num_classes):
    value = (value or '').strip().lower()
    if value in ('all', '-1'):
        return list(range(num_classes))

    selected = set()
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            start, end = (int(part) for part in item.split('-', 1))
            if start > end:
                start, end = end, start
            selected.update(
                class_id for class_id in range(start, end + 1)
                if 0 <= class_id < num_classes
            )
        else:
            class_id = int(item)
            if 0 <= class_id < num_classes:
                selected.add(class_id)
    return sorted(selected)


def checkpoint_for(
    method,
    dataset_name,
    model_name,
    forget_classes,
    lr,
    base_dir,
):
    """Resolve the single- or multi-class checkpoint used in this project."""
    forget_classes = sorted(set(int(c) for c in forget_classes))
    if not forget_classes:
        raise ValueError('At least one forget class is required.')

    if method == 'original':
        return os.path.join(
            base_dir,
            'test_pretrained_model',
            f'{dataset_name}_{model_name}_original_model.pth',
        )

    if len(forget_classes) == 1:
        forget_tag = f'forgetcls{forget_classes[0]}'
    else:
        forget_tag = f'forget{len(forget_classes)}'

    if method == 'retrained':
        return os.path.join(
            base_dir,
            'test_pretrained_model',
            f'{dataset_name}_{model_name}_retrain_{forget_tag}_model.pth',
        )

    return os.path.join(
        base_dir,
        f'{dataset_name}_{model_name}_{forget_tag}',
        method,
        f'lr{lr}',
        'ckpt_best_by_aus.pth',
    )


def load_checkpoint_model(
    checkpoint_path,
    model_name,
    dataset_name,
    num_classes,
    device,
):
    # Load on CPU first to avoid an unnecessary duplicate allocation on GPU.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model_state_dict', 'model'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break

        if not isinstance(state_dict, dict):
            raise TypeError(
                f'Unsupported checkpoint object: {type(checkpoint).__name__}'
            )

        state_dict = {
            key.removeprefix('module.'): value
            for key, value in state_dict.items()
        }
        # models.get_model() historically forces vit-b-16 onto CUDA. Build
        # that architecture directly so --device cpu and cuda:N both work.
        if model_name == 'vit-b-16':
            model = ViT_16_mod(n_classes=num_classes)
        else:
            model = get_model(
                model_name,
                dataset_name,
                num_classes,
                use_pretrained=False,
            )
        model.load_state_dict(state_dict)

    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def extract_dataset_features(model, loader, device):
    """Extract the exact input to the model's active linear classifier."""
    model.eval()
    classifier = get_active_classifier(model)
    features = []
    labels = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        batch_features = extract_features(
            model,
            images,
            classifier=classifier,
        )
        features.append(batch_features.float().cpu())
        labels.append(targets.long().cpu())

    return torch.cat(features), torch.cat(labels)


def masked_accuracy(predictions, targets, forget_classes):
    forget_mask = torch.zeros_like(targets, dtype=torch.bool)
    for class_id in forget_classes:
        forget_mask |= targets.eq(class_id)
    retain_mask = ~forget_mask

    def accuracy(mask):
        if not bool(mask.any()):
            return float('nan')
        return float(
            predictions[mask].eq(targets[mask]).float().mean().item() * 100.0
        )

    return accuracy(retain_mask), accuracy(forget_mask)


@torch.no_grad()
def evaluate_original_head(
    model,
    loader,
    device,
    forget_classes,
):
    predictions = []
    targets = []
    model.eval()
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.long().cpu())
    return masked_accuracy(
        torch.cat(predictions),
        torch.cat(targets),
        forget_classes,
    )


def train_linear_probe(
    train_features,
    train_targets,
    num_classes,
    device,
    seed,
    learning_rate,
    momentum,
    weight_decay,
    batch_size,
    patience,
    max_epochs,
    val_ratio,
    use_cosine,
):
    """
    Follow CMF_Unlearning/evaluation/linear_prob.py:
    fresh linear layer, SGD, 10% validation, and validation early stopping.
    """
    torch.manual_seed(seed)
    feature_dim = train_features.shape[1]
    classifier = nn.Linear(feature_dim, num_classes).to(device)

    full_dataset = TensorDataset(train_features, train_targets)
    validation_size = max(1, int(len(full_dataset) * val_ratio))
    training_size = len(full_dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    training_dataset, validation_dataset = random_split(
        full_dataset,
        [training_size, validation_size],
        generator=generator,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
        )
        if use_cosine else None
    )
    criterion = nn.CrossEntropyLoss()

    best_validation_accuracy = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        classifier.train()
        for batch_features, batch_targets in training_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(classifier(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_targets in validation_loader:
                predictions = classifier(
                    batch_features.to(device)
                ).argmax(dim=1).cpu()
                correct += int(predictions.eq(batch_targets).sum().item())
                total += int(batch_targets.numel())
        validation_accuracy = correct / max(1, total)

        improved = validation_accuracy > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f'Epoch {epoch:03d}/{max_epochs}: '
                f'validation accuracy={100.0 * validation_accuracy:.2f}%'
            )

        if scheduler is not None:
            scheduler.step()
        if epochs_without_improvement >= patience:
            print(
                f'Early stopping at epoch {epoch}; '
                f'best epoch={best_epoch}.'
            )
            break

    if best_state is None:
        raise RuntimeError('Linear probe training produced no checkpoint.')
    classifier.load_state_dict(best_state)
    return classifier, best_epoch, 100.0 * best_validation_accuracy


@torch.no_grad()
def evaluate_probe(
    classifier,
    features,
    targets,
    device,
    forget_classes,
    batch_size,
):
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    all_targets = []
    classifier.eval()
    for batch_features, batch_targets in loader:
        batch_predictions = classifier(
            batch_features.to(device, non_blocking=True)
        ).argmax(dim=1)
        predictions.append(batch_predictions.cpu())
        all_targets.append(batch_targets)
    return masked_accuracy(
        torch.cat(predictions),
        torch.cat(all_targets),
        forget_classes,
    )


def resolve_device(requested_device):
    device = torch.device(requested_device)
    if device.type != 'cuda':
        return device
    if not torch.cuda.is_available():
        print('CUDA is unavailable; falling back to CPU.')
        return torch.device('cpu')
    if (
        device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise ValueError(
            f'Requested {device}, but only '
            f'{torch.cuda.device_count()} CUDA device(s) are visible.'
        )
    return device


def save_result(result, args):
    output_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f'{args.dataset}_{args.model_name}.csv',
    )
    fieldnames = [
        'forget_classes',
        'num_forget_classes',
        'seed',
        'output_acc_r_test',
        'output_acc_f_test',
        'linear_probe_acc_r_test',
        'linear_probe_acc_f_test',
        'linear_probe_gain_r_test',
        'linear_probe_gain_f_test',
        'best_validation_acc',
        'best_epoch',
        'feature_dim',
        'train_samples',
        'test_samples',
    ]

    header_needed = not os.path.isfile(output_path)
    with open(output_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if header_needed:
            writer.writeheader()
        writer.writerow({
            key: (
                f'{value:.4f}'
                if isinstance(value, float)
                else value
            )
            for key, value in result.items()
        })
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Apply the CMF_Unlearning paper linear-probe evaluation to '
            'this repository\'s unlearned checkpoints.'
        )
    )
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument(
        '--base-dir',
        default=str(EXPS_DIR),
    )
    parser.add_argument(
        '--method',
        required=True,
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
    )
    parser.add_argument(
        '--dataset',
        required=True,
        choices=list(NUM_CLASSES),
    )
    parser.add_argument('--model', dest='model_name', default='resnet18')
    parser.add_argument(
        '--forget',
        required=True,
        help='Forgotten classes, for example 25,58 or 0-4.',
    )
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--probe-batch-size', type=int, default=128)
    parser.add_argument('--feature-batch-size', type=int, default=256)
    parser.add_argument('--probe-lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Defaults to 20 for CIFAR-10 and 150 otherwise.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default='linear_probe')
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    num_classes = NUM_CLASSES[args.dataset]
    forget_classes = parse_forget_arg(args.forget, num_classes)
    if not forget_classes:
        raise ValueError(
            f'No valid classes in --forget={args.forget!r} for '
            f'{args.dataset}.'
        )

    device = resolve_device(args.device)
    print(f'Using device: {device}')
    if args.checkpoint:
        checkpoint_path = os.path.expanduser(args.checkpoint)
    else:
        checkpoint_path = checkpoint_for(
            args.method,
            args.dataset,
            args.model_name,
            forget_classes,
            args.lr,
            os.path.expanduser(args.base_dir),
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    print(f'Checkpoint: {checkpoint_path}')
    print(f'Forgotten classes: {forget_classes}')
    model = load_checkpoint_model(
        checkpoint_path,
        args.model_name,
        args.dataset,
        num_classes,
        device,
    )

    transform_train, transform_test = get_transforms(
        args.dataset,
        args.model_name,
        wo_dataaug=False,
    )
    train_dataset, test_dataset = get_dataset(
        args.dataset,
        transform_train,
        transform_test,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    output_retain, output_forget = evaluate_original_head(
        model,
        test_loader,
        device,
        forget_classes,
    )
    print(
        f'Original head: retain={output_retain:.2f}%, '
        f'forget={output_forget:.2f}%'
    )

    print('Extracting frozen train features...')
    train_features, train_targets = extract_dataset_features(
        model,
        train_loader,
        device,
    )
    print('Extracting frozen test features...')
    test_features, test_targets = extract_dataset_features(
        model,
        test_loader,
        device,
    )
    print(
        f'Feature shapes: train={tuple(train_features.shape)}, '
        f'test={tuple(test_features.shape)}'
    )

    max_epochs = args.max_epochs
    if max_epochs is None:
        max_epochs = 20 if args.dataset == 'cifar10' else 150

    probe, best_epoch, best_validation_accuracy = train_linear_probe(
        train_features,
        train_targets,
        num_classes,
        device,
        args.seed,
        args.probe_lr,
        args.momentum,
        args.weight_decay,
        args.probe_batch_size,
        args.patience,
        max_epochs,
        args.val_ratio,
        use_cosine=args.dataset == 'tiny_imagenet',
    )
    probe_retain, probe_forget = evaluate_probe(
        probe,
        test_features,
        test_targets,
        device,
        forget_classes,
        args.probe_batch_size,
    )

    print('\nLinear-probe result:')
    print(
        f'  retain accuracy: {output_retain:.2f}% '
        f'-> {probe_retain:.2f}%'
    )
    print(
        f'  forget accuracy: {output_forget:.2f}% '
        f'-> {probe_forget:.2f}%'
    )
    print(
        f'  best validation accuracy: '
        f'{best_validation_accuracy:.2f}% at epoch {best_epoch}'
    )

    result = {
        'forget_classes': ','.join(map(str, forget_classes)),
        'num_forget_classes': len(forget_classes),
        'seed': args.seed,
        'output_acc_r_test': output_retain,
        'output_acc_f_test': output_forget,
        'linear_probe_acc_r_test': probe_retain,
        'linear_probe_acc_f_test': probe_forget,
        'linear_probe_gain_r_test': probe_retain - output_retain,
        'linear_probe_gain_f_test': probe_forget - output_forget,
        'best_validation_acc': best_validation_accuracy,
        'best_epoch': best_epoch,
        'feature_dim': train_features.shape[1],
        'train_samples': len(train_targets),
        'test_samples': len(test_targets),
    }
    output_path = save_result(result, args)
    print(f'CSV saved to: {output_path}')


if __name__ == '__main__':
    main()
