# Source-Free Class Relearning: Diagnosing Forgetting in Class Unlearning

This repository implements Source-Free Relearning Analysis (SFRA), an audit for measuring how readily an unlearned class can be recovered by updating only the released classifier head with synthetic feature-space probes. SFRA does not require access to the original training data or real samples from the forget class. The repository contains the training, unlearning, relearning, analysis, and visualization code used to reproduce the paper's experiments.

The repository supports:

- **Datasets:** `cifar10`, `cifar100`, `tiny_imagenet`
- **Backbones:** `resnet18`, `vit-b-16`, `swin-t`
- **Unlearning methods:**  
  `retrained`, `random_label`, `finetune`, `gradient_ascent`,  
  `boundary_shrink`, `delete`,
  `l2ul_adv`, `salun`, `scrub`, `bad_teacher`, `neggrad_plus`

- Training original models
- Retrain-from-scratch baselines
- Single-class and multi-class unlearning
- Our source-free class relearning method
- Ablations and t-SNE visualizations

## Notation

- $N$ is the number of accepted synthetic candidate embeddings collected per
  retained class before confidence-based selection. In the commands below,
  this is controlled by `--tpr`, `--per_class`, or `--generated_per_class`,
  depending on the script.
- $M$ is the number of synthetic probes selected per retained class. When the
  retain and forget counts differ, we use $M_r$ and $M_f$; these correspond to
  `--retain_per_class` and `--forget_per_class`, respectively.

### Relearning Score

RS is the relearning score. Let $\mathcal{A}^{t\text{-}un}_r$ and
$\mathcal{A}^{t\text{-}un}_f$ denote the retain and forget test accuracies
after unlearning, and let $\mathcal{A}^{t\text{-}re}_r$ and
$\mathcal{A}^{t\text{-}re}_f$ denote the corresponding test accuracies after
relearning. With all accuracies normalized to $[0,1]$, the retain-preservation
and forget-recovery terms are

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

Thus, $\mathrm{RS}\in[0,1]$; a high value indicates strong forget-class
recovery with limited retain-accuracy degradation. For method $m$, audit
variant $v$, and forget class $c$, the matched-control score is

```math
\Delta\mathrm{RS}^{(v)}_{m,c}
=\mathrm{RS}^{(v)}_{m,c}
-\mathrm{RS}^{(v)}_{\mathrm{retrained},c}.
```

RS measures absolute source-free recoverability, whereas
$\Delta\mathrm{RS}$ measures recoverability relative to the retrained model
matched to the same forget class.

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

All core scripts use the same portable path policy: an explicit command-line
argument takes priority, followed by an environment variable, followed by a
default location under the current user's home directory.

```bash
export SFRA_DATA_DIR="$HOME/data"
export SFRA_EXPS_DIR="$HOME/classification/exps"
```

On the system used for the paper, these expand to
`/export/livia/home/vision/Zdehghani/data` and
`/export/livia/home/vision/Zdehghani/classification/exps`. Other users can set
the variables to their own locations without modifying the source code.
CIFAR-10 and CIFAR-100 are downloaded automatically under `SFRA_DATA_DIR`.
TinyImageNet should have the following layout:

```text
$SFRA_DATA_DIR/TinyImageNet/
  train/
  val/
  test/
```

Individual commands can still override these defaults using their applicable
`--data_dir`, `--base-dir`, `--base_dir`, `--ckpt_dir`, or `--exps_dir`
argument. Generated result directories are resolved relative to the repository
unless an explicit output directory is provided.

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
CUDA_VISIBLE_DEVICES=0 python source_free_relearning_single_class.py \
  --method bad_teacher \
  --model_name resnet18 \
  --dataset cifar10 \
  --lr 1e-3 \
  --epochs 500 \
  --forget 0,1,2,3,4,5,6,7,8,9 \
  --retain_per_class 500 \
  --forget_per_class 500 \
  --tpr 500000
```


**multi-class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python source_free_relearning_multi_class.py \
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

The multi-class generator supports the 2-class and 5-vs-10-class table
formats. Each format reads from its existing results directory and writes the
generated table back to that directory:

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

## 10. Appendix / Plot and Table Generation Scripts

These scripts are post-processing utilities used to reproduce appendix figures and tables from saved checkpoints and result CSV files. Most scripts assume that the corresponding unlearned checkpoints and result folders have already been generated.

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

This script validates the approximation used in the one-step margin analysis. It compares the exact weighted update term with the unweighted synthetic mean approximation and generates appendix figures/tables for the coefficient validation experiment.

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

This script loads an unlearned checkpoint and visualizes real test samples together with selected synthetic probes. It saves separate t-SNE figures for the pre-classifier feature space and the classifier-head logit space.

### Synthesis-ablation RS table (Appendix O)

```bash
python plots_tables/generate_synthesis_ablation_rs_table.py \
  --root results_synthesis_ablation \
  --methods retrained bad_teacher neggrad_plus delete scrub boundary_shrink finetune gradient_ascent random_label l2ul_adv salun \
  --dataset cifar10 \
  --model_name resnet18 \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --exclude_strategies dist=uniform__forget=low_confidence__retain=high_confidence \
  --bold_best
```

This script reads the results produced by the synthesis-strategy ablation and creates a LaTeX table with forget classes as columns. It reports the relearning score (RS) for different synthesis strategies.

### Synthetic--real alignment analysis (Appendix D)

This post-hoc analysis uses only the low-confidence synthetic forget probes from
the existing source-free relearning procedure. It does not train or modify a
model and does not construct random or high-confidence controls.

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

Then run the full CIFAR-10/ResNet-18 analysis with the settings used by the
existing relearning results:

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


### Gaussian vs. Uniform embedding-distribution table (Appendix O)

```bash
python plots_tables/generate_sampling_distribution_rs_table.py \
  --dataset cifar10 \
  --model_name resnet18 \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --bold_best
```

This script compares Gaussian and Uniform embedding distributions. It combines Gaussian results from `results_single_class` with Uniform results from `results_synthesis_ablation`.

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

This script evaluates the unlearned model on real test samples and measures how confidently retain classes are assigned to correctly classified retain samples and to forget-class samples that are predicted as retain classes.

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

This script aggregates the per-forget-class confidence CSVs into one compact table. For each forget class, it reports the weighted average confidence of correctly classified retain samples, the weighted average confidence of forget-class samples assigned to retain classes, and their confidence gap.
By default, the aggregated CSV and LaTeX table are written to
`results/<method>/`; for this example, the table is saved as
`results/bad_teacher/bad_teacher_cifar10_resnet18_weighted_confidence.tex`.


### Synthesis-strategy ablation runner (Appendices O and Q)

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

### Uncertainty-score ablation (Appendix P)

This experiment compares Softmax confidence, predictive entropy, and energy
for selecting probes from the same accepted Gaussian candidate pool. It runs
all CIFAR-10 forget classes (`0`--`9`) for three independent audit seeds
(`0`, `1`, and `2`) and trains the classifier head for up to 500 epochs. One
launcher processes all enabled unlearning methods sequentially for a selected
architecture.

Run the three architectures on separate GPUs with:

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

After all methods finish successfully for an architecture, the launcher
automatically generates its per-class RS table and corresponding CSV. The
following commands are only needed to regenerate the tables from existing
results without rerunning the experiments:

```bash
python plots_tables/generate_uncertainty_per_class_rs_table.py --model resnet18
python plots_tables/generate_uncertainty_per_class_rs_table.py --model vit-b-16
python plots_tables/generate_uncertainty_per_class_rs_table.py --model swin-t
```

The table cells report RS mean $\pm$ standard deviation across the three audit
seeds, and the final `Avg.` column reports the mean RS across forget classes
and seeds. The generated `.tex` and `.csv` tables are saved in the same
uncertainty-ablation results directory.

### Retain--forget trajectory analysis (Appendix I)

This analysis trains the released classifier head on one fixed Gaussian
synthetic dataset and evaluates real retain and forget accuracy at every
epoch. It saves one trajectory CSV per method and seed, together with the
retain--forget trade-off summaries.

For CIFAR-10/ResNet-18 with forget class~7, run:

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

The recoverability-map script compares SFRA RS with matched-control
$\Delta$RS across datasets, architectures, unlearning methods, and forgotten
classes. It reads the generated single-class appendix tables and creates one
map for each architecture.

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
