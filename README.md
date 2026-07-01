# Source-Free Class Relearning: Diagnosing Forgetting in Class Unlearning


It supports:

- **Datasets:** `cifar10`, `cifar100`, `tiny_imagenet`
- **Backbones:** `resnet18`, `vit-s-16`, `vit-b-16`, `swin-t`, `vgg16`
- **Unlearning methods (`--method`):**  
  `retrained`, `random_label`, `finetune`, `gradient_ascent`,  
  `boundary_shrink`, `boundary_expand`, `delete`, 
  `l2ul_adv`, `salun`, `scrub`, `bad_teacher`, `neggrad_plus`

- Training original models
- Retrain-from-scratch baselines
- Single-class and multi-class unlearning
- Class relearning method
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

**Single class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py  --classes "0,1,2,3,4,5,6,7,8,9" --method pass  --dataset_name cifar10 --model_name resnet18 --retrain_from_scratch
```

**Multi class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py  --forget_class 2   --method pass  --dataset_name cifar10  	--model_name resnet18 --retrain_from_scratch
```

---

## 4. Apply unelarning methods:

**Single class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets cifar10 	--model-name resnet18	--forget-classes 0 1 2 3 4 5 6 7 8 9
```

**multi class unlearning:**
```bash
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets cifar10 	--model-name resnet18	--forget-set 1 6
```

Evaluate the unlearn model:

**Single class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python eval_unlearned_model.py   --dataset_name cifar10  	--model_name resnet18  	--method bad_teacher   		--forget_id 0 		--unlearn_rate 1e-03  --exps_dir ~/classification/exps   --batch_size 256   --num_workers 8
```


**multi class unlearning:**
```bash
CUDA_VISIBLE_DEVICES=0 python eval_unlearned_model.py   --dataset_name cifar100  	--model_name renset18   --method bad_teacher  		--forget_set 25 58    	--unlearn_rate 1e-03  --exps_dir ~/classification/exps   --batch_size 128   --num_workers 8
```

---

## 5. Run our class relearning method:

**Single class setting:**
```bash
CUDA_VISIBLE_DEVICES=0 python source_free_relearning_singleclass.py 	--method bad_teacher  --model_name resnet18 	--dataset cifar10	    	--lr 1e-3 	--epochs 500  --forget 0,1,2,3,4,5,6,7,8,9    	--retain_per_class 500   	--forget_per_class 500   --tpr 500000
```


**multi class unlearning:**
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
CUDA_VISIBLE_DEVICES=0 python -m ablation.build_synth_pool.py  --dataset cifar10 --model_name resnet18 --method scrub  --per_class 500000   --forget_class 7   --lr 1e-3
CUDA_VISIBLE_DEVICES=0 python -m ablation.relearning_singleclass_ablation.py    --pool_path synth_pools/cifar10_resnet18_bad_teacher_fg7_lr0.001_pool_500000perclass_10cls_emb512_seed0.pt  --pool_take_per_class 500000  --dataset cifar10   --model_name resnet18   --method delete   --retain_per_class 500   --forget_per_class 500  --lr 0.001  --forget 7   --save_synth_dir results/bad_teacher/synth_pt  --save_real_dir  results/bad_teacher/real_pt  --save_ckpt_dir results/bad_teacher/
CUDA_VISIBLE_DEVICES=0 python tsne_framework.py 	--dataset cifar10 --model_name resnet18  	--method bad_teacher 	--forget_class 7 --lr 0.001  --split test --autoname
```

tsne of the unlearned model for the synthetic embeddings and the real embeddings:
```bash
CUDA_VISIBLE_DEVICES=0 python plots_tables/tsne_real_gaussian_probes.py
   --method bad_teacher   --dataset cifar10   --model_name resnet18   --lr 0.001   --forget_class 7   --generated_per_class 500000   --retain_per_class 180   --forget_per_class 20   --real_per_class 180   --perplexity 30   --tsne_iterations 1500   --seed 0
```



CUDA_VISIBLE_DEVICES=2 python run_pra_on_single_checkpoint.py   --dataset cifar100   --model resnet18   --method boundary_shrink   --lr 1e-08   --forget 0,20,40,60,80   --base-dir /export/livia/home/vision/Zdehghani/classification/exps   --n-percent 100.0   --num-samples-per-class 5   --pra-metric cosine   --max-retain-drop 1.0   --attack-split train   --support-seed 0   --device cuda




CUDA_VISIBLE_DEVICES=2 python run_linear_probe_single_checkpoint.py   --dataset tiny_imagenet   --model vit-b-16   --method delete   --forget 0,40,80,120,160   --lr 0.001

CUDA_VISIBLE_DEVICES=2 python run_linear_probe_multi_checkpoint.py   --dataset cifar10   --model resnet18   --method scrub   --forget 1,6   --lr 0.001


python plots_tables/synthesis_ablation_class_table_rs_only.py   --root results_synthesis_ablation   --methods bad_teacher neggrad_plus delete scrub boundary_shrink finetune gradient_ascent random_label l2ul_adv salun --dataset cifar10   --model_name resnet18   --classes 0 1 2 3 4 5 6 7 8 9   --bold_best