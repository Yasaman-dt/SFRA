# Source-Free Class Revival: Diagnosing Forgetting in Machine Unlearning


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
- Class revival method
- Ablations and t-SNE visualizations

---

## 1. Environment Setup

Tested with:

- **Python:** 3.10.13

Recommended setup with conda:

# Create and activate environment
conda create  -y -n torch_env python=3.10
conda activate torch_env

# Install dependencies
python -m pip install -r requirements.txt


## 2. Train the original model from scratch

CUDA_VISIBLE_DEVICES=0 python main.py  --method pass  --dataset_name tiny_imagenet  --model_name resnet18   --train_from_scratch   --use_pretrained


## 3. Apply Retrain-from-scratch baseline without the forget class:

# Single-class:

# CIFAR-10
CUDA_VISIBLE_DEVICES=0 python main.py  --classes "0,1,2,3,4,5,6,7,8,9" --method pass  --dataset_name cifar10 --model_name resnet18 --retrain_from_scratch
# CIFAR-100
CUDA_VISIBLE_DEVICES=0 python main.py  --classes "0,20,40,60,80"   --method pass   --dataset_name cifar100   --model_name resnet18   --retrain_from_scratch   --use_pretrained
# Tiny-ImageNet
CUDA_VISIBLE_DEVICES=0 python main.py --classes "0,40,80,120,160" --method pass --dataset_name tiny_imagenet --model_name resnet18 --retrain_from_scratch   --use_pretrained

# Multi_class:
CUDA_VISIBLE_DEVICES=0 python main.py  --forget_class 2   --method pass  --dataset_name cifar10  	--model_name resnet18 --retrain_from_scratch


## 4. Apply unelarning methods:

# single_class:
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets cifar10 	--model-name resnet18	--forget-classes 0 1 2 3 4 5 6 7 8 9
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets cifar100 	--model-name resnet18	--forget-classes 0 20 40 60 80
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets tiny_imagenet 	--model-name resnet18  	--forget-classes 0 40 80 120 160

# multi_class:
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets cifar10 	--model-name resnet18	--forget-set 1 6
CUDA_VISIBLE_DEVICES=0 python test_original_model.py --datasets cifar100 	--model-name resnet18	--forget-set 25 58


# evaluate the unlearn model:
# single_class:
CUDA_VISIBLE_DEVICES=0 python eval_unlearned_model.py   --dataset_name cifar10  	--model_name resnet18  	--method bad_teacher   		--forget_id 0 		--unlearn_rate 1e-03  --exps_dir ~/classification/exps   --batch_size 256   --num_workers 8

# multi_class:
CUDA_VISIBLE_DEVICES=0 python eval_unlearned_model.py   --dataset_name cifar100  	--model_name renset18   --method bad_teacher  		--forget_set 25 58    	--unlearn_rate 1e-03  --exps_dir ~/classification/exps   --batch_size 128   --num_workers 8


## 5. Run our class revival method:

# single_class:
CUDA_VISIBLE_DEVICES=0 python revival_singleclass.py 	--method bad_teacher  --model_name resnet18 	--dataset cifar10	    	--lr 1e-3 	--epochs 500  --forget 0,1,2,3,4,5,6,7,8,9    	--retain_per_class 500   	--forget_per_class 500   --tpr 500000
CUDA_VISIBLE_DEVICES=0 python revival_singleclass.py 	--method bad_teacher  --model_name resnet18 	--dataset cifar100	    	--lr 1e-3 	--epochs 500  --forget 0,20,40,60,80    	--retain_per_class 100   	--forget_per_class 100   --tpr 100000
CUDA_VISIBLE_DEVICES=0 python revival_singleclass.py 	--method bad_teacher  --model_name resnet18 	--dataset tiny_imagenet		--lr 1e-3 	--epochs 200  --forget 0,40,80,120,160  	--retain_per_class 25   	--forget_per_class 25 	 --tpr 50000

# multi_class:
CUDA_VISIBLE_DEVICES=0 python revival_multiclass.py  --dataset cifar10   	--model resnet18   --method bad_teacher  --lr 2e-2   --epochs 200   --forget 1,6   	--tpr 500000   	--retain_per_class 500   --forget_per_class 500
CUDA_VISIBLE_DEVICES=0 python revival_multiclass.py  --dataset cifar100   	--model resnet18   --method bad_teacher	 --lr 2e-2   --epochs 200   --forget 25,58   	--tpr 500000   	--retain_per_class 200   --forget_per_class 200



## 6. ablation on different M and N:
CUDA_VISIBLE_DEVICES=0 python build_synth_pool.py  			--dataset cifar10 	--model_name resnet18 	--method delete  --per_class 500000   --forget_class 9   --lr 1e-3
CUDA_VISIBLE_DEVICES=0 python revival_singleclass_ablation.py    	--pool_path synth_pools/cifar10_resnet18_bad_teacher_fg9_lr0.001_pool_500000perclass_10cls_emb512_seed0.pt  --pool_take_per_class 500000  --dataset cifar10   --model_name resnet18   --method bad_teacher   --retain_per_class 500   --forget_per_class 500  --lr 0.001  --forget 9

## 7. tsne plots:
# tsne of the unlearned model:
CUDA_VISIBLE_DEVICES=0 python tsne_plot1.py 		--dataset cifar10 --model_name resnet18  	--method bad_teacher 	--forget_class 7 	--lr 0.001 	--split test 	--autoname
CUDA_VISIBLE_DEVICES=0 python tsne_plot2.py   		--dataset cifar10 --model_name resnet18   	--method bad_teacher 	--forget_class 7 	--lr 0.001  	--forget "7"   	--split test --max_per_class 500 --perplexity 30   --autoname

## tsne of the unlearned model and reival model:
CUDA_VISIBLE_DEVICES=0 python build_synth_pool.py  --dataset cifar10 --model_name resnet18 --method scrub  --per_class 500000   --forget_class 7   --lr 1e-3
CUDA_VISIBLE_DEVICES=0 python revival_singleclass_ablation.py    --pool_path synth_pools/cifar10_resnet18_bad_teacher_fg7_lr0.001_pool_500000perclass_10cls_emb512_seed0.pt  --pool_take_per_class 500000  --dataset cifar10   --model_name resnet18   --method delete   --retain_per_class 500   --forget_per_class 500  --lr 0.001  --forget 7   --save_synth_dir results/bad_teacher/synth_pt  --save_real_dir  results/bad_teacher/real_pt  --save_ckpt_dir results/bad_teacher/
CUDA_VISIBLE_DEVICES=0 python tsne_framework.py 	--dataset cifar10 --model_name resnet18  	--method bad_teacher 	--forget_class 7 --lr 0.001  --split test --autoname

