# DUE

This repo is for [DUE: Dynamic Uncertainty-Aware Explanation Supervision via 3D Imputation](https://arxiv.org/abs/2403.10831) 

<img src="https://github.com/AlexQilong/DUE/blob/main/assets/framework_overview.png" alt="due_overview" style="width:70%;">

## Setup

Please refer to `requirements.txt`

## Implementation

1. Train a [diffusion model](https://github.com/voletiv/mcvd-pytorch) for slice interpolation
2. Interpolate annotation slices repeatly using the diffusion model to estimate uncertainty
3. Train a [VAE](https://github.com/XiYe20/NPVP) to directly estimate uncertainty
4. Set paths and run:   
   `python main.py --model "due" --dataset $DATASET --attention_weight 1 --seed 0`

## Deployment

Below are screenshots illustrating the deployment of DUE, using lung nodule classification as an example:

### 1. Visual Annotation Labeling Interface
The screenshots below display the interface for labeling visual annotations. Radiologists can annotate images by drawing on them, generating a binary matrix of the focus area. This process contributes to enhancing the quality of model explanations.

<img src="https://github.com/AlexQilong/DUE/blob/main/assets/screenshot_cancer_1.png" style="width:100%;">
<img src="https://github.com/AlexQilong/DUE/blob/main/assets/screenshot_cancer_2.png" style="width:100%;">

### 2. Model Selection Interface
Here is the interface for selecting the model, where users can choose from trained model checkpoints.

<img src="https://github.com/AlexQilong/DUE/blob/main/assets/screenshot_model_select.png" style="width:80%;">
