# DUE

This repo is for [DUE: Dynamic Uncertainty-Aware Explanation Supervision via 3D Imputation](https://arxiv.org/abs/2403.10831) 

<img src="https://github.com/AlexQilong/DUE/blob/main/assets/framework_overview.png" alt="due_overview" style="width:50%;">

## Setup

Please refer to `requirements.txt`

## Training

1. Train a [diffusion model](https://github.com/voletiv/mcvd-pytorch) for slice interpolation
2. Interpolate annotation slices repeatly using the diffusion model to estimate uncertainty 
3. Train a [VAE](https://github.com/XiYe20/NPVP) to directly estimate uncertainty
4. Set paths and run:   
   `python main.py --model "due" --dataset $DATASET --attention_weight 1 --seed 0`
