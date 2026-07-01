import argparse

from relearning_baselines.run_linear_probe_on_checkpoint import (
    NUM_CLASSES,
    main as run_linear_probe,
    parse_forget_arg,
)


METHODS = [
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
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Run the linear-probe audit on one checkpoint that unlearned '
            'multiple classes simultaneously.'
        )
    )
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument(
        '--dataset',
        required=True,
        choices=list(NUM_CLASSES),
    )
    parser.add_argument('--model', dest='model_name', default='resnet18')
    parser.add_argument(
        '--forget',
        required=True,
        help='Classes forgotten together, e.g. 25,58,38,23,96.',
    )
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument(
        '--base-dir',
        default='/export/livia/home/vision/Zdehghani/classification/exps',
    )
    parser.add_argument('--probe-batch-size', type=int, default=128)
    parser.add_argument('--feature-batch-size', type=int, default=256)
    parser.add_argument('--probe-lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default='linear_probe_multi')
    args = parser.parse_args()

    forget_classes = parse_forget_arg(
        args.forget,
        NUM_CLASSES[args.dataset],
    )
    if len(forget_classes) < 2:
        raise ValueError(
            'The multi-class script requires at least two valid classes '
            'in --forget.'
        )

    probe_arguments = [
        '--method', args.method,
        '--dataset', args.dataset,
        '--model', args.model_name,
        '--forget', ','.join(map(str, forget_classes)),
        '--lr', str(args.lr),
        '--base-dir', args.base_dir,
        '--probe-batch-size', str(args.probe_batch_size),
        '--feature-batch-size', str(args.feature_batch_size),
        '--probe-lr', str(args.probe_lr),
        '--momentum', str(args.momentum),
        '--weight-decay', str(args.weight_decay),
        '--val-ratio', str(args.val_ratio),
        '--patience', str(args.patience),
        '--seed', str(args.seed),
        '--num-workers', str(args.num_workers),
        '--device', args.device,
        '--output-dir', args.output_dir,
    ]
    if args.checkpoint is not None:
        probe_arguments.extend(['--checkpoint', args.checkpoint])
    if args.max_epochs is not None:
        probe_arguments.extend(['--max-epochs', str(args.max_epochs)])

    print(
        'Running one simultaneously unlearned checkpoint for classes: '
        f'{forget_classes}'
    )
    run_linear_probe(probe_arguments)


if __name__ == '__main__':
    main()
