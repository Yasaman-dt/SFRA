# Source-Free Class Revival: Diagnosing Forgetting in Machine Unlearning


It supports:

- **Datasets:** `cifar10`, `cifar100`, `tiny_imagenet`
- **Backbones:** `resnet18`, `vit-s-16`, `vit-b-16`, `swin-t`, `vgg16`
- **Unlearning methods (`--method`):**  
  `retrained`, `random_label`, `finetune`, `gradient_ascent`,  
  `boundary_shrink`, `boundary_expand`, `delete`, 
  `l2ul_adv`, `salun`, `scrub`, `bad_teacher`, `neggrad_plus`

  Both **single-class** and **multi-class** forgetting are supported.


---

## 1. Environment Setup

Tested with:

- **Python:** 3.10.13

Recommended setup with conda:

```bash
# Create and activate environment
conda create  -y -n torch_env python=3.10
conda activate torch_env

# Install dependencies
python -m pip install -r requirements.txt


