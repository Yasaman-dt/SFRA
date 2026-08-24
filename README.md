# Source-Free Class Relearning: Diagnosing Forgetting in Class Unlearning


It supports:

- **Datasets:** `cifar10`, `cifar100`, `tiny_imagenet`
- **Backbones:** `resnet18`, `vit-s-16`, `vit-b-16`, `swin-t`, `vgg16`
- **Unlearning methods:**  
  `retrained`, `random_label`, `finetune`, `gradient_ascent`,  
  `boundary_shrink`, `boundary_expand`, `delete`, 
  `l2ul_adv`, `salun`, `scrub`, `bad_teacher`, `neggrad_plus`

- Training original models
- Retrain-from-scratch baselines
- Single-class and multi-class unlearning
- Our source-free class relearning method
- Ablations and t-SNE visualizations

---

## 1. Environment Setup

Tested with:
- **Python:** 3.10.13

Create and activate environment:
```bash
conda create  -y -n torch_env python=3.10
conda activate torch_env
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

---

## 2. Train the original model from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python main.py  --method pass  --dataset_name tiny_imagenet  --model_name resnet18   --train_from_scratch   --use_pretrained
```

---

## 3. Apply Retrain-from-scratch baseline without the forget class:

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py  --classes "0,1,2,3,4,5,6,7,8,9" --method pass  --dataset_name cifar10 --model_name resnet18 --retrain_from_scratch
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py  --forget_class 2   --method pass  --dataset_name cifar10  	--model_name resnet18 --retrain_from_scratch
```

---

## Test original model for different settings:

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets cifar10 	--model-name resnet18	--forget-classes 0 1 2 3 4 5 6 7 8 9
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets cifar10 	--model-name resnet18	--forget-set 1 6
```

---

## 4. Apply unlearning methods:

**single-class setting:**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py  --dataset_name cifar10  	--classes "0,1,2,3,4,5,6,7,8,9" 	--model_name resnet18  --retain_data True --method bad_teacher  	--unlearn_rate 1e-3  	--description lr_1e-3
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=1 python main.py  --dataset_name cifar10 		--forget_class 2 	--model_name resnet18   --method gradient_ascent  --retain_data True  --unlearn_rate 1e-5  --description lr_1e-5
```

---


Evaluate the unlearned model:

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python eval_unlearned_model.py   --dataset_name cifar10  	--model_name resnet18  	--method bad_teacher   		--forget_id 0 		--unlearn_rate 1e-03  --exps_dir ~/classification/exps   --batch_size 256   --num_workers 8
```


**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python eval_unlearned_model.py   --dataset_name cifar100  	--model_name resnet18   --method bad_teacher  		--forget_set 25 58    	--unlearn_rate 1e-03  --exps_dir ~/classification/exps   --batch_size 128   --num_workers 8
```

---

## 5. Run our class relearning method:

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python source_free_relearning_singleclass.py 	--method bad_teacher  --model_name resnet18 	--dataset cifar10	    	--lr 1e-3 	--epochs 500  --forget 0,1,2,3,4,5,6,7,8,9    	--retain_per_class 500   	--forget_per_class 500   --tpr 500000
```


**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python source_free_relearning_multiclass.py  --dataset cifar10   	--model resnet18   --method bad_teacher  --lr 2e-2   --epochs 200   --forget 1,6   	--tpr 500000   	--retain_per_class 500   --forget_per_class 500
```

---

## 6. ablation on different M and N:
```bash
CUDA_VISIBLE_DEVICES=0 python -m ablation.build_synth_pool  			--dataset cifar10 	--model_name resnet18 	--method bad_teacher  --per_class 500000   --forget_class 9   --lr 1e-3
CUDA_VISIBLE_DEVICES=0 python -m ablation.relearning_singleclass_ablation    	--pool_path synth_pools/cifar10_resnet18_bad_teacher_fg9_lr0.001_pool_500000perclass_10cls_emb512_seed0.pt  --pool_take_per_class 500000  --dataset cifar10   --model_name resnet18   --method bad_teacher   --retain_per_class 500   --forget_per_class 500  --lr 0.001  --forget 9
```

---

## 7. tsne plots:

tsne of the unlearned model:
```bash
cd plots_tables
CUDA_VISIBLE_DEVICES=0 python tsne_plot1.py 		--dataset cifar10 --model_name resnet18  	--method bad_teacher 	--forget_class 7 	--lr 0.001 	--split test 	--autoname
CUDA_VISIBLE_DEVICES=0 python tsne_plot2.py   		--dataset cifar10 --model_name resnet18   	--method bad_teacher 	--forget_class 7 	--lr 0.001  	--forget "7"   	--split test --max_per_class 500 --perplexity 30   --autoname
```

tsne of the unlearned model and relearned model:
```bash
CUDA_VISIBLE_DEVICES=0 python -m ablation.build_synth_pool  --dataset cifar10 --model_name resnet18 --method scrub  --per_class 500000   --forget_class 7   --lr 1e-3
CUDA_VISIBLE_DEVICES=0 python -m ablation.relearning_singleclass_ablation    --pool_path synth_pools/cifar10_resnet18_bad_teacher_fg7_lr0.001_pool_500000perclass_10cls_emb512_seed0.pt  --pool_take_per_class 500000  --dataset cifar10   --model_name resnet18   --method delete   --retain_per_class 500   --forget_per_class 500  --lr 0.001  --forget 7   --save_synth_dir results/bad_teacher/synth_pt  --save_real_dir  results/bad_teacher/real_pt  --save_ckpt_dir results/bad_teacher/
CUDA_VISIBLE_DEVICES=0 python tsne_framework.py 	--dataset cifar10 --model_name resnet18  	--method bad_teacher 	--forget_class 7 --lr 0.001  --split test --autoname
```

## Source-Dependent Baselines
We evaluate two source-dependent baselines: PRA and linear probing. Unlike our source-free relearning audit, these baselines use real data from the original task.

### PRA baseline

PRA is evaluated using real support samples from the forgotten classes. For single-class unlearning checkpoints, we run:

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=2 python -m relearning_baselines.run_pra_on_single_checkpoint \
  --dataset cifar100 \
  --model resnet18 \
  --method boundary_shrink \
  --lr 1e-08 \
  --forget 0,10,20,30,40,50,60,70,80,90 \
  --n-percent 100.0 \
  --num-samples-per-class 5 \
  --pra-metric cosine \
  --max-retain-drop 1.0 \
  --attack-split train \
  --support-seed 0 
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=2 python -m relearning_baselines.run_pra_on_multi_checkpoint \
  --dataset cifar100 \
  --model resnet18 \
  --method finetune \
  --forget 25,58 \
  --lr 0.02
```


### Linear-Probe Baseline
The linear-probe baseline evaluates how much class information remains in the representation of the unlearned model.

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=2 python -m relearning_baselines.run_linear_probe_single_checkpoint \
  --dataset tiny_imagenet \
  --model vit-b-16 \
  --method delete \
  --forget 0,20,40,60,80,100,120,140,160,180 \
  --lr 0.001
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=2 python -m relearning_baselines.run_linear_probe_multi_checkpoint \
  --dataset cifar10 \
  --model resnet18 \
  --method scrub \
  --forget 1,6 \
  --lr 0.001
```


---

## 8. Appendix / Plot and Table Generation Scripts

These scripts are post-processing utilities used to reproduce appendix figures and tables from saved checkpoints and result CSV files. Most scripts assume that the corresponding unlearned checkpoints and result folders have already been generated.

### Coefficient approximation validation

```bash
python plots_tables/probe_coefficient_validation.py \
  --method bad_teacher \
  --dataset cifar10 \
  --model_name resnet18 \
  --lr 0.001 \
  --forget_class 7 \
  --generated_per_class 500000 \
  --selected_per_class 500 \
  --bootstrap_repetitions 300
```

This script validates the approximation used in the one-step margin analysis. It compares the exact weighted update term with the unweighted synthetic mean approximation and generates appendix figures/tables for the coefficient validation experiment.

### Real and synthetic probe t-SNE visualization

```bash
CUDA_VISIBLE_DEVICES=0 python plots_tables/tsne_real_gaussian_probes.py \
  --method bad_teacher \
  --dataset cifar10 \
  --model_name resnet18 \
  --lr 0.001 \
  --forget_class 7 \
  --generated_per_class 500000 \
  --retain_per_class 500 \
  --forget_per_class 500 \
  --real_per_class 500
```

This script loads an unlearned checkpoint and visualizes real test samples together with selected synthetic probes. It saves separate t-SNE figures for the pre-classifier feature space and the classifier-head logit space.

### Synthesis-ablation RS table

```bash
python plots_tables/synthesis_ablation_class_table_rs_only.py \
  --root results_synthesis_ablation \
  --methods retrained bad_teacher neggrad_plus delete scrub boundary_shrink finetune gradient_ascent random_label l2ul_adv salun \
  --dataset cifar10 \
  --model_name resnet18 \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --exclude_strategies dist=uniform__forget=low_confidence__retain=high_confidence \
  --bold_best
```

This script reads the results produced by the synthesis-strategy ablation and creates a LaTeX table with forget classes as columns. It reports the relearning score (RS) for different synthesis strategies.

### Synthetic--real alignment analysis

This post-hoc analysis uses only the low-confidence synthetic forget probes from
the existing source-free relearning procedure. It does not train or modify a
model and does not construct random or high-confidence controls.

Run a small one-class sanity check first:

```bash
CUDA_VISIBLE_DEVICES=0 python plots_tables/synthetic_real_alignment.py \
  --methods bad_teacher delete scrub retrained \
  --forget_classes 0 \
  --seeds 0 \
  --N 1000 --M 100 \
  --skip_retain_pass \
  --output_dir results_single_class/analysis/alignment_analysis_smoke
```

Then run the full CIFAR-10/ResNet-18 analysis with the settings used by the
existing relearning results:

```bash
CUDA_VISIBLE_DEVICES=0 python plots_tables/synthetic_real_alignment.py \
  --dataset cifar10 \
  --model resnet18 \
  --methods bad_teacher delete scrub retrained \
  --forget_classes 0 1 2 3 4 5 6 7 8 9 \
  --seeds 0 \
  --N 500000 --M 500 \
  --output_dir results_single_class/analysis/alignment_analysis
```

Omitting `--skip_retain_pass` preserves the original generator's RNG-consuming
retain-pool pass before constructing the actual low-confidence forget probes.


### Gaussian vs. Uniform embedding-distribution table

```bash
python plots_tables/sampling_distribution_rs_table.py \
  --dataset cifar10 \
  --model_name resnet18 \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --bold_best
```

This script compares Gaussian and Uniform embedding distributions. It combines Gaussian results from `results_single_class` with Uniform results from `results_synthesis_ablation`.

### Confidence of forget-class assignments

First generate per-forget-class confidence statistics:

```bash
CUDA_VISIBLE_DEVICES=0 python plots_tables/retain_forget_assigned_confidence.py \
  --method bad_teacher \
  --dataset cifar10 \
  --model_name resnet18 \
  --lr 0.001 \
  --forget_classes 0 1 2 3 4 5 6 7 8 9
```

This script evaluates the unlearned model on real test samples and measures how confidently retain classes are assigned to correctly classified retain samples and to forget-class samples that are predicted as retain classes.

### Aggregated confidence table across forget classes

```bash
python plots_tables/aggregate_confidence_by_forget_class.py \
  --root results_single_class/analysis/confidence_tables/bad_teacher \
  --method bad_teacher \
  --dataset cifar10 \
  --model_name resnet18 \
  --lr 0.001 \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --allow_any_lr
```

This script aggregates the per-forget-class confidence CSVs into one compact table. For each forget class, it reports the weighted average confidence of correctly classified retain samples, the weighted average confidence of forget-class samples assigned to retain classes, and their confidence gap.


### Synthesis-strategy ablation runner

The following command runs the Laplace-only synthesis ablation for SCRUB on
CIFAR-10 with ResNet-18. It uses the same candidate pool for selecting the
high-confidence retain probes and low-confidence forget probes. The Gaussian
baseline is skipped because its existing results are reused.

```bash
CUDA_VISIBLE_DEVICES=2 python -m ablation.synthesis_strategy_ablation \
  --dataset cifar10 \
  --model_name resnet18 \
  --generated_per_class 500000 \
  --retain_per_class 500 \
  --forget_per_class 500 \
  --sample_batch_size 4096 \
  --train_batch_size 256 \
  --epochs 200 \
  --head_lr 0.01 \
  --weight_decay 0.0001 \
  --retain_floor_frac 0.90 \
  --grid one_factor \
  --distributions laplace \
  --skip_gaussian_baseline \
  --forget_selections low_confidence \
  --retain_selections high_confidence \
  --uncertainty_scores softmax \
  --seeds 0 \
  --method scrub \
  --lr 0.001 \
  --forget_classes 0 1 2 3 4 5 6 7 8 9
```
