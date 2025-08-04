# Anonymous Submission

This repository contains code and experiments related to our anonymous submission to AAAI 2026 (under blind review).  
Author identities have been anonymized for peer review.

## Prerequisites
```
conda create --name cps-prompt python=3.12
conda activate cps-prompt
pip install -r requirements.txt
```

## Running Experiments

Examples:

CIFAR-100
  ```
  python main.py --dataset seq-cifar100-224 --model cps-prompt --lr 1e-3 --batch_size 16
  ```
ImageNet-R
  ```
  python main.py --dataset seq-imagenet-r --model cps-prompt --lr 1e-3 --batch_size 16
  ```
CUB200
  ```
  python main.py --dataset seq-cub200 --model cps-prompt --lr 1e-3 --batch_size 16
  ```

## Acknowledgement
Note: This code is based on the [Mammoth](https://github.com/aimagelab/mammoth) and [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt).