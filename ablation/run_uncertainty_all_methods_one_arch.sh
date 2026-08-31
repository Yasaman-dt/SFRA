#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 GPU MODEL [OUTPUT_DIR]" >&2
  echo "Models: resnet18, vit-b-16, swin-t" >&2
  exit 2
fi

gpu="$1"
model="$2"
output_dir="${3:-results_uncertainty_ablation_cifar10_all_classes_500ep}"

extra_args=()
if [[ "${FORCE:-0}" == "1" ]]; then
  extra_args+=(--force)
fi

common_args=(
  --dataset cifar10
  --model_name "$model"
  --generated_per_class 500000
  --retain_per_class 500
  --forget_per_class 500
  --sample_batch_size 4096
  --train_batch_size 256
  --eval_batch_size 256
  --epochs 500
  --patience 500
  --head_lr 0.01
  --weight_decay 0.0001
  --retain_floor_frac 0.90
  --grid one_factor
  --distributions gaussian
  --forget_selections low_confidence
  --retain_selections high_confidence
  --uncertainty_scores softmax entropy energy
  --seeds 0 1 2
  --output_dir "$output_dir"
)

run_classes() {
  local method="$1"
  local lr="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" python -m ablation.synthesis_strategy_ablation \
    --method "$method" \
    --lr "$lr" \
    --forget_classes "$@" \
    "${common_args[@]}" \
    "${extra_args[@]}"
}

case "$model" in
  resnet18)
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
    ;;
  vit-b-16)
    jobs=(
      "retrained:0"
      "finetune:0.02"
      "gradient_ascent:1e-05"
      "neggrad_plus:0.5"
      "random_label:0.001"
      "l2ul_adv:1e-05"
      "scrub:0.0009"
      "bad_teacher:0.1"
      "salun:0.001"
      "delete:0.001"
    )
    ;;
  swin-t)
    jobs=(
      "retrained:0"
      "finetune:0.02"
      "gradient_ascent:1e-06"
      "neggrad_plus:0.1"
      "random_label:1e-06"
      "boundary_shrink:1e-07"
      "l2ul_adv:1e-06"
      "scrub:0.0001"
      "bad_teacher:0.01"
      "salun:0.0001"
      "delete:0.001"
    )
    ;;
  *)
    echo "Unsupported model: $model" >&2
    exit 2
    ;;
esac

for job in "${jobs[@]}"; do
  method="${job%%:*}"
  lr="${job#*:}"
  echo "[start] model=$model method=$method lr=$lr gpu=$gpu"
  if [[ "$model" == "resnet18" && "$method" == "bad_teacher" ]]; then
    # The selected checkpoints use lr=0.2 for forget classes 2 and 4 and
    # lr=0.001 for the remaining CIFAR-10 classes.
    run_classes "$method" 0.001 0 1 3 5 6 7 8 9
    run_classes "$method" 0.2 2 4
  else
    run_classes "$method" "$lr" 0 1 2 3 4 5 6 7 8 9
  fi
  echo "[done] model=$model method=$method"
done
