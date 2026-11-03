# Source-Free Class Relearning: Diagnosing Forgetting in Class Unlearning

Source-Free Relearning Analysis (SFRA) audits class unlearning by recovering forgotten classes through classifier-head updates on synthetic feature probes. This repository reproduces the paper's training, unlearning, relearning, and analysis experiments.

The repository supports:

- **Datasets:** `cifar10`, `cifar100`, `tiny_imagenet`
- **Backbones:** `resnet18`, `vit-b-16`, `swin-t`
- **Unlearning methods:**  
  `retrained`, `random_label`, `finetune`, `gradient_ascent`,  
  `boundary_shrink`, `delete`,
  `l2ul_adv`, `salun`, `scrub`, `bad_teacher`, `neggrad_plus`

Supports original training, retraining, single- and multi-class unlearning, SFRA, ablations, and visualization.

## Notation

- $N$: accepted synthetic candidates per retained class (`--tpr`, `--per_class`, or `--generated_per_class`).
- $M$: selected probes per retained class; $M_r$ and $M_f$ denote `--retain_per_class` and `--forget_per_class`.

### Relearning Score

RS combines retain preservation and forget recovery, using accuracies before ($un$) and after ($re$) relearning, normalized to $[0,1]$:

```math
\begin{aligned}
R_r &= 1-\max\!\left(
0,\mathcal{A}^{t\text{-}un}_r-\mathcal{A}^{t\text{-}re}_r
\right), \\
R_f &= \max\!\left(
0,\mathcal{A}^{t\text{-}re}_f-\mathcal{A}^{t\text{-}un}_f
\right).
\end{aligned}
```

Their harmonic mean defines

```math
\mathrm{RS}=
\begin{cases}
\displaystyle\frac{2R_rR_f}{R_r+R_f}, & R_r+R_f>0, \\
0, & \text{otherwise}.
\end{cases}
```

Higher RS indicates stronger recovery with less retain loss. `RS_val` selects the epoch within `--max_retain_drop_frac` (default: 10% of epoch-0 retain accuracy); `RS_test` and `RS_synth_train` are reporting metrics. The matched-control score compares method $m$ with retraining for audit variant $v$ and forget class $c$:

```math
\Delta\mathrm{RS}^{(v)}_{m,c}
=\mathrm{RS}^{(v)}_{m,c}
-\mathrm{RS}^{(v)}_{\mathrm{retrained},c}.
```

RS measures absolute recoverability; $\Delta\mathrm{RS}$ measures recovery relative to matched retraining.

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

### Data and checkpoint paths

Path priority: command-line argument, environment variable, then home-directory default.

```bash
export SFRA_DATA_DIR="$HOME/data"
export SFRA_EXPS_DIR="$HOME/classification/exps"
```

CIFAR-10/100 download automatically. Set the variables for your system; TinyImageNet requires:

```text
$SFRA_DATA_DIR/TinyImageNet/
  train/  # class subfolders
  val/    # labeled held-out evaluation set, used as test
```

Unlearning and baselines use a class-stratified 90% training / 10% validation split (seed `42`). Use the same split seed across scripts. SFRA trains on synthetic probes; PRA draws real support samples from the 90% subset (`--support-seed` controls sampling), and linear probing trains a fresh head on that subset. Validation selects checkpoints, epochs, or PRA interpolation strength; final evaluation uses the official CIFAR test set or TinyImageNet `val/`.

Override paths with the script's `--data_dir`, `--base-dir`, `--base_dir`, `--ckpt_dir`, or `--exps_dir`. Relative result paths resolve from the repository.

---

## 2. Train the original model from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --method pass \
  --dataset_name tiny_imagenet \
  --model_name resnet18 \
  --train_from_scratch \
  --use_pretrained
```

---

## 3. Apply the Retrain-from-Scratch Baseline Without the Forget Class

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --classes "0,1,2,3,4,5,6,7,8,9" \
  --method pass \
  --dataset_name cifar10 \
  --model_name resnet18 \
  --retrain_from_scratch
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --forget_class 2 \
  --method pass \
  --dataset_name cifar10 \
  --model_name resnet18 \
  --retrain_from_scratch
```

---

## 4. Test the Original Model for Different Settings

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python test_original_model.py \
  --datasets cifar10 \
  --model-name resnet18 \
  --forget-classes 0 1 2 3 4 5 6 7 8 9
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python test_original_model.py \
  --datasets cifar10 \
  --model-name resnet18 \
  --forget-set 1 6
```

---

## 5. Apply Unlearning Methods

**single-class setting:**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_name cifar10 \
  --classes "0,1,2,3,4,5,6,7,8,9" \
  --model_name resnet18 \
  --retain_data True \
  --method bad_teacher \
  --unlearn_rate 1e-3 \
  --description lr_1e-3
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
  --dataset_name cifar10 \
  --forget_class 2 \
  --model_name resnet18 \
  --method gradient_ascent \
  --retain_data True \
  --unlearn_rate 1e-5 \
  --description lr_1e-5
```

CIFAR-10/100 runs report train, validation, and test metrics; validation AUS selects the checkpoint.

---


Evaluate the unlearned model:

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python test_unlearned_model.py \
  --dataset_name cifar10 \
  --model_name resnet18 \
  --method bad_teacher \
  --forget_id 0 \
  --unlearn_rate 1e-03 \
  --exps_dir "$SFRA_EXPS_DIR" \
  --batch_size 256 \
  --num_workers 8
```


**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python test_unlearned_model.py \
  --dataset_name cifar100 \
  --model_name resnet18 \
  --method bad_teacher \
  --forget_set 25 58 \
  --unlearn_rate 1e-03 \
  --exps_dir "$SFRA_EXPS_DIR" \
  --batch_size 128 \
  --num_workers 8
```

---

## 6. Run Our Class Relearning Method

**single-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python SFRA_single_class.py \
  --method bad_teacher \
  --model resnet18 \
  --dataset cifar10 \
  --lr 1e-3 \
  --epochs 500 \
  --forget 0 \
  --retain_per_class 500 \
  --forget_per_class 500 \
  --tpr 500000
```
Use `--forget all` to run all CIFAR-10 forget classes.


**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python SFRA_multi_class.py \
  --dataset cifar10 \
  --model resnet18 \
  --method bad_teacher \
  --lr 2e-2 \
  --epochs 200 \
  --forget 1,6 \
  --tpr 500000 \
  --retain_per_class 500 \
  --forget_per_class 500
```

### Generate the summary tables (Main paper and Appendix J)

Generate the single-class summary tables:

```bash
python plots_tables/summary_tables/generate_single_class_summary_table.py
```

The generated architecture-level tables are saved under
`results_single_class/tables/single_class_summary/`.

Multi-class tables are written to their input results directories:

```bash
# Generate both multi-class summary-table formats
python plots_tables/summary_tables/generate_multi_class_summary_tables.py \
  --setting all

# Generate only the 2-class summary table
python plots_tables/summary_tables/generate_multi_class_summary_tables.py \
  --setting 2

# Generate only the 5-vs-10-class comparison table
python plots_tables/summary_tables/generate_multi_class_summary_tables.py \
  --setting 5-10
```

---

## 7. Ablation on Different M and N (Main paper and Appendix H)
```bash
CUDA_VISIBLE_DEVICES=0 python -m ablation.build_synthetic_probe_pool \
  --dataset cifar10 \
  --model_name resnet18 \
  --method bad_teacher \
  --per_class 500000 \
  --forget_class 9 \
  --lr 1e-3

CUDA_VISIBLE_DEVICES=0 python -m ablation.run_single_class_relearning_ablation \
  --pool_path synth_pools/cifar10_resnet18_bad_teacher_fg9_lr0.001_pool_500000perclass_10cls_emb512_seed0.pt \
  --pool_take_per_class 500000 \
  --dataset cifar10 \
  --model_name resnet18 \
  --method bad_teacher \
  --retain_per_class 500 \
  --forget_per_class 500 \
  --lr 0.001 \
  --forget 9
```

---

## 8. t-SNE Plots (Main paper and Appendix K)

tsne of the unlearned model:
```bash
CUDA_VISIBLE_DEVICES=0 python plots_tables/tsne_plots/plot_unlearned_feature_tsne.py \
  --dataset cifar10 \
  --model_name resnet18 \
  --method bad_teacher \
  --forget_class 7 \
  --lr 0.001 \
  --split test \
  --autoname

CUDA_VISIBLE_DEVICES=0 python plots_tables/tsne_plots/plot_unlearned_logit_tsne.py \
  --dataset cifar10 \
  --model_name resnet18 \
  --method bad_teacher \
  --forget_class 7 \
  --lr 0.001 \
  --forget "7" \
  --split test \
  --max_per_class 500 \
  --perplexity 30 \
  --autoname
```

tsne of the unlearned model and relearned model:
```bash
CUDA_VISIBLE_DEVICES=0 python -m ablation.build_synthetic_probe_pool \
  --dataset cifar10 \
  --model_name resnet18 \
  --method scrub \
  --per_class 500000 \
  --forget_class 7 \
  --lr 1e-3

CUDA_VISIBLE_DEVICES=0 python -m ablation.run_single_class_relearning_ablation \
  --pool_path synth_pools/cifar10_resnet18_scrub_fg7_lr0.001_pool_500000perclass_10cls_emb512_seed0.pt \
  --pool_take_per_class 500000 \
  --dataset cifar10 \
  --model_name resnet18 \
  --method scrub \
  --retain_per_class 500 \
  --forget_per_class 500 \
  --lr 0.001 \
  --forget 7 \
  --save_synth_dir results/scrub/synth_pt \
  --save_real_dir results/scrub/real_pt \
  --save_ckpt_dir results/scrub/

CUDA_VISIBLE_DEVICES=0 python plots_tables/tsne_plots/plot_unlearned_vs_relearned_tsne.py \
  --dataset cifar10 \
  --model_name resnet18 \
  --method scrub \
  --forget_class 7 \
  --lr 0.001 \
  --split test \
  --autoname
```

## 9. Source-Dependent Baselines (Main paper and Appendices J and R)
PRA and linear probing use real task data.

### PRA baseline

PRA uses forgotten-class support samples from the training subset and selects interpolation strength on validation data.

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
  --support-seed 0 \
  --seed 42
```

**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=2 python -m relearning_baselines.run_pra_on_multi_checkpoint \
  --dataset cifar100 \
  --model resnet18 \
  --method finetune \
  --forget 25,58 \
  --lr 0.02 \
  --seed 42
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

## 10. Appendix / Plot and Table Generation Scripts

Generate figures and tables from existing checkpoints and result CSVs.

### Appendix-to-code map

| Appendix | Draft section | Primary code |
| --- | --- | --- |
| A | Proof of Proposition 1 | Analytical proof; no experiment script required |
| B | Hyperparameter Settings | `plots_tables/generate_training_hyperparameter_tables.py` |
| C | Computational Cost and Efficiency | `analysis/benchmark_probe_generation.py`; `plots_tables/generate_probe_timing_architecture_table.py` |
| D | Post-hoc Synthetic--Real Alignment | `analysis/synthetic_real_alignment.py`; `plots_tables/aggregate_alignment_results.py`; `plots_tables/generate_alignment_per_class_table.py` |
| E | Empirical Assessment of the Margin Approximation | `analysis/probe_coefficient_validation.py` |
| F | Confidence of Forget Class Assignments | `analysis/retain_forget_assigned_confidence.py`; `plots_tables/generate_forget_class_confidence_table.py` |
| G | SFRA Without the Released Forget Class Output Row | `ablation/run_forget_row_ablation_fg7_all_methods.sh`; `plots_tables/generate_forget_row_ablation_table.py` |
| H | Main paper (Fig. 5) and Sensitivity to the Number of Synthetic Probes | `ablation/build_synthetic_probe_pool.py`; `ablation/run_single_class_relearning_ablation.py`; `plots_tables/plot_probe_count_ablation.py` |
| I | Retain--Forget Accuracy Trade-off | `ablation/retain_forget_tradeoff.py`; `plots_tables/plot_rs_vs_accuracy_change.py`; `plots_tables/plot_retain_forget_tradeoff.py` |
| J | Main paper (Tables 1–2) and Additional Single- and Multi-Class SFRA Results | `plots_tables/summary_tables/generate_single_class_summary_table.py`; `plots_tables/summary_tables/generate_multi_class_summary_tables.py` |
| K | Main paper (Fig. 4) and Geometric Interpretation of Synthetic Boundary Probes | `plots_tables/tsne_plots/plot_real_vs_gaussian_probe_tsne.py` |
| L | Main paper (Fig. 2) and RS Distribution Across Forget Classes | `plots_tables/plot_rs_violin_distributions.py` |
| M | Main paper (Fig. 3) and Per-Class RS Heatmaps | `plots_tables/plot_per_class_rs_heatmaps.py` |
| N | Absolute and Excess Recoverability | `plots_tables/plot_rs_recoverability_map.py` |
| O | Sampling Distribution Ablation | `ablation/synthesis_strategy_ablation.py`; `plots_tables/generate_sampling_distribution_rs_table.py` |
| P | Uncertainty-Score Ablation | `ablation/run_uncertainty_all_methods_one_arch.sh`; `plots_tables/generate_uncertainty_per_class_rs_table.py` |
| Q | Effect of Gaussian Support on SFRA | `ablation/synthesis_strategy_ablation.py` |
| R | Detailed Per-Class Results and Linear Separability | `relearning_baselines/run_linear_probe_single_checkpoint.py`; `plots_tables/generate_single_class_variant_table.py` |

The Appendix R per-forget-class `.tex` tables are written to
`results_single_class/tables/single_class_variant_by_forget/` by default.
The probe-count sensitivity figures for Main Figure 5 and Appendix H are saved
as PNG files under `results_single_class/plots/probe_count_ablation/`.

### Coefficient approximation validation (Appendix E)

```bash
python analysis/probe_coefficient_validation.py \
  --method bad_teacher \
  --dataset cifar10 \
  --model_name resnet18 \
  --lr 0.001 \
  --forget_class 7 \
  --generated_per_class 500000 \
  --selected_per_class 500 \
  --bootstrap_repetitions 300
```

Compares the exact weighted update with the synthetic-mean approximation and generates validation figures and tables.

### Real and synthetic probe t-SNE visualization (Appendix K)

```bash
CUDA_VISIBLE_DEVICES=0 python plots_tables/tsne_plots/plot_real_vs_gaussian_probe_tsne.py \
  --method bad_teacher \
  --dataset cifar10 \
  --model_name resnet18 \
  --lr 0.001 \
  --forget_class 7 \
  --generated_per_class 500000 \
  --retain_per_class 25 \
  --forget_per_class 25 \
  --real_per_class 225
```

Plots real test samples and selected synthetic probes in feature and logit spaces.

### Synthesis-ablation RS table (Appendix O)

```bash
python plots_tables/generate_synthesis_ablation_rs_table.py \
  --root results_synthesis_ablation/results_synthesis_ablation_resnet18 \
  --methods retrained bad_teacher neggrad_plus delete scrub boundary_shrink finetune gradient_ascent random_label l2ul_adv salun \
  --dataset cifar10 \
  --model_name resnet18 \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --exclude_strategies dist=uniform__forget=low_confidence__retain=high_confidence \
  --bold_best
```

Creates a per-forget-class LaTeX RS table from synthesis-ablation results.

### Synthetic--real alignment analysis (Appendix D)

Analyzes existing low-confidence synthetic forget probes without modifying the model.

Run a small one-class sanity check first:

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/synthetic_real_alignment.py \
  --methods bad_teacher delete scrub retrained \
  --forget_classes 0 \
  --seeds 0 \
  --N 1000 --M 100 \
  --skip_retain_pass \
  --output_dir results_single_class/analysis/alignment_analysis_smoke
```

Full CIFAR-10/ResNet-18 analysis:

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/synthetic_real_alignment.py \
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


### Sampling-distribution RS table (Appendix O)

```bash
python plots_tables/generate_sampling_distribution_rs_table.py \
  --dataset cifar10 \
  --model_name resnet18 \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --bold_best
```

Compares Gaussian results from `results_single_class` with Uniform/Laplace synthesis results. Saves `.tex` and `.csv` tables under `results_synthesis_ablation/results_synthesis_ablation_resnet18/`.

### Confidence of forget-class assignments (Appendix F)

First generate per-forget-class confidence statistics:

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/retain_forget_assigned_confidence.py \
  --method bad_teacher \
  --dataset cifar10 \
  --model_name resnet18 \
  --lr 0.001 \
  --forget_classes 0 1 2 3 4 5 6 7 8 9
```

Measures retain-assignment confidence for correctly classified retain samples and forget samples predicted as retain.

### Aggregated confidence table across forget classes (Appendix F)

```bash
python plots_tables/generate_forget_class_confidence_table.py \
  --root results_single_class/analysis/confidence_tables/bad_teacher \
  --method bad_teacher \
  --dataset cifar10 \
  --model_name resnet18 \
  --lr 0.001 \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --allow_any_lr
```

Aggregates per-class confidence CSVs into weighted retain/forget confidence and gap tables under `results/<method>/`.

### Synthesis-strategy ablation runner (Appendices O and Q)

Runs Laplace synthesis for SCRUB on CIFAR-10/ResNet-18, sharing candidates for retain/forget selection and reusing Gaussian baseline results.

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

Unless `--output_dir` is supplied, synthesis-ablation runs are grouped by
architecture under:

```text
results_synthesis_ablation/
  results_synthesis_ablation_resnet18/
  results_synthesis_ablation_vit-b-16/
```

### Uncertainty-score ablation (Appendix P)

Compares Softmax, entropy, and energy selection on shared Gaussian candidates across all CIFAR-10 forget classes, three seeds, and up to 500 epochs. Launch each architecture on a separate GPU:

```bash
mkdir -p logs

# GPU 0: all ResNet-18 methods
nohup bash ablation/run_uncertainty_all_methods_one_arch.sh \
  0 resnet18 \
  > logs/uncertainty_resnet18_all_methods.log 2>&1 &

# GPU 1: all ViT-B/16 methods
nohup bash ablation/run_uncertainty_all_methods_one_arch.sh \
  1 vit-b-16 \
  > logs/uncertainty_vit_b16_all_methods.log 2>&1 &

# GPU 2: all Swin-T methods
nohup bash ablation/run_uncertainty_all_methods_one_arch.sh \
  2 swin-t \
  > logs/uncertainty_swin_t_all_methods.log 2>&1 &
```

By default, all run-level and summary CSV files are saved under:

```text
results_uncertainty_ablation_cifar10_all_classes_500ep/
```

An optional third launcher argument overrides this output directory:

```bash
bash ablation/run_uncertainty_all_methods_one_arch.sh \
  0 resnet18 custom_uncertainty_results
```

Launchers generate tables automatically. To regenerate from existing results:

```bash
python plots_tables/generate_uncertainty_per_class_rs_table.py --model resnet18
python plots_tables/generate_uncertainty_per_class_rs_table.py --model vit-b-16
python plots_tables/generate_uncertainty_per_class_rs_table.py --model swin-t
```

Tables report RS mean $\pm$ standard deviation across seeds and an overall `Avg.`, saved as `.tex` and `.csv` in the results directory.

### Retain--forget trajectory analysis (Appendix I)

Tracks real retain/forget accuracy while training the head on fixed Gaussian probes. Validation selects the reported epoch; trajectory CSVs include final test metrics. Example for CIFAR-10/ResNet-18, forget class 7:

```bash
CUDA_VISIBLE_DEVICES=0 python -m ablation.retain_forget_tradeoff \
  --dataset cifar10 \
  --model_name resnet18 \
  --forget_class 7 \
  --methods bad_teacher delete gradient_ascent random_label salun \
            retrained finetune neggrad_plus l2ul_adv \
  --generated_per_class 500000 \
  --retain_per_class 500 \
  --forget_per_class 500 \
  --epochs 500 \
  --seeds 0 1 2 \
  --output_dir results_retain_forget_tradeoff/cifar10_resnet18_fg7
```

The per-seed trajectories are written as:

```text
results_retain_forget_tradeoff/cifar10_resnet18_fg7/
  METHOD/trajectory_seed0.csv
  METHOD/trajectory_seed1.csv
  METHOD/trajectory_seed2.csv
```

After generating matching CIFAR-10 and CIFAR-100 trajectories, create the
combined RS-versus-retain-change plot with:

```bash
python plots_tables/plot_rs_vs_accuracy_change.py \
  --cifar10-dir results_retain_forget_tradeoff/cifar10_resnet18_fg7 \
  --cifar100-dir results_retain_forget_tradeoff/cifar100_resnet18_fg0 \
  --x-metric retain \
  --legend first-upper-left \
  --output results_single_class/plots/rs_vs_retain_accuracy_change.png
```

### RS recoverability maps (Appendix N)

Plots SFRA RS against matched-control $\Delta$RS per architecture using single-class appendix tables.

```bash
python plots_tables/plot_rs_recoverability_map.py
```

By default, the figures and supporting CSV files are saved under:

```text
results_single_class/plots/rs_recoverability_outputs/
```

Use `--architectures` to select particular backbones or `--output-dir` to
override the destination, for example:

```bash
python plots_tables/plot_rs_recoverability_map.py \
  --architectures resnet18 vit-b-16 swin-t \
  --output-dir results_single_class/plots/rs_recoverability_outputs
```

---

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@misc{dehghani2026sourcefreeclassrelearningdiagnosing,
  title         = {Source-Free Class Relearning: Diagnosing Forgetting in Class Unlearning},
  author        = {Zahra Dehghani and Pablo Piantanida and Mohammadhadi Shateri},
  year          = {2026},
  eprint        = {2609.02018},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.02018}
}
```

