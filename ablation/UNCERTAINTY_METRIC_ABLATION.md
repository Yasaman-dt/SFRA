# MSP, entropy, and energy probe-selection ablation

Run the complete CIFAR-10/ResNet-18 experiment with:

```bash
bash ablation/run_uncertainty_metric_ablation.sh
```

The command evaluates forget classes 0--9 for the matched retrained control,
Bad Teacher, DELETE, SCRUB, Negative Gradient+, and Finetune. It fixes the
Gaussian candidate pool at `N=500000`, selects `M=500` retain and forget probes
per retain class, and changes only `--uncertainty_metric` (`msp`, `entropy`, or
`energy`). The feature extractor remains frozen and only the classifier head is
updated.

For a single diagnostic run, use for example:

```bash
python -m ablation.synthesis_strategy_ablation \
  --method bad_teacher --dataset cifar10 --model_name resnet18 --lr 0.001 \
  --forget_class 0 --grid one_factor --distributions gaussian \
  --forget_selections low_confidence --retain_selections high_confidence \
  --uncertainty_metric msp entropy energy \
  --generated_per_class 500000 --retain_per_class 500 --forget_per_class 500 \
  --epochs 200 --head_lr 0.01 --weight_decay 0.0001 \
  --train_batch_size 256 --seeds 0
```

After all method and retrained runs finish, aggregate them with:

```bash
python plots_tables/uncertainty_metric_ablation.py \
  --methods bad_teacher delete scrub neggrad_plus finetune
```

Outputs are written under `results_uncertainty_ablation/`: per-class and
aggregated CSVs, raw and aggregated overlap@M CSVs, a LaTeX table, RS/Delta-RS
plots, and a short automatic text summary. Each Delta-RS value is matched to a
retrained run using the same forget class, seed, and uncertainty metric.

Energy uses `E=-logsumexp(logits)` at temperature 1. Larger (less negative)
values and higher predictive entropy are treated as more uncertain. For MSP,
lower assigned-class probability is more uncertain. Thus high-uncertainty
probes become relabeled forget probes and low-uncertainty probes become retain
probes for every metric.
