#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 GPU [OUTPUT_DIR]" >&2
  exit 2
fi

gpu="$1"
output_dir="${2:-results_forget_row_ablation_cifar10_fg7}"
forget_class=7
seeds=(0 1 2)

# Learning rates of the selected CIFAR-10/ResNet-18 class-7 checkpoints.
jobs=(
  "retrained:0"
  "finetune:0.02"
  "gradient_ascent:5e-05"
  "neggrad_plus:0.5"
  "random_label:1e-07"
  "boundary_shrink:1e-08"
  "l2ul_adv:1e-05"
  "scrub:0.001"
  "bad_teacher:0.001"
  "salun:0.001"
  "delete:0.001"
)

common_args=(
  --dataset cifar10
  --model_name resnet18
  --forget "$forget_class"
  --epochs 500
  --retain_per_class 500
  --forget_per_class 500
  --tpr 500000
  --retain_floor_frac 0.90
  --rs_patience 500
)

for run_seed in "${seeds[@]}"; do
  seed_output_dir="$output_dir/seed_$run_seed"
  random_head_seed=$((42 + run_seed))

  for job in "${jobs[@]}"; do
    method="${job%%:*}"
    lr="${job#*:}"

    echo "[start] seed=$run_seed method=$method condition=released_head lr=$lr"
    CUDA_VISIBLE_DEVICES="$gpu" python source_free_relearning_singleclass.py \
      --method "$method" \
      --lr "$lr" \
      --seed "$run_seed" \
      --output-dir "$seed_output_dir" \
      "${common_args[@]}"
    echo "[done] seed=$run_seed method=$method condition=released_head"

    echo "[start] seed=$run_seed method=$method condition=random_forget_row lr=$lr"
    CUDA_VISIBLE_DEVICES="$gpu" python source_free_relearning_singleclass.py \
      --method "$method" \
      --lr "$lr" \
      --seed "$run_seed" \
      --output-dir "$seed_output_dir" \
      --randomize_forget_row \
      --random_head_seed "$random_head_seed" \
      "${common_args[@]}"
    echo "[done] seed=$run_seed method=$method condition=random_forget_row"
  done
done

echo "All class-$forget_class forget-row ablations completed: $output_dir"
