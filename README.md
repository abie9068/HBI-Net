# HBI-Net

This repository contains project materials for:

**HBI-Net: Hierarchical Bitemporal Interaction Network for Directional Algal Bloom Change Detection in Multispectral Imagery**

It includes method-level descriptions, training settings reported in the manuscript, and reference interfaces for the model and dataset.

## Contents

- `configs/`: configuration summaries and dataset interface templates
- `hbi_net/`: model and dataset reference interfaces
- `scripts/`: command entry interfaces
- `docs/`: method summary
- `requirements.txt`: framework-level dependencies

## Task Definition

The model takes paired Sentinel-2 multispectral image patches as input:

- `T1`: earlier-time 10-band patch
- `T2`: later-time 10-band patch

The output is a pixel-level three-class map:

1. unchanged
2. bloom appearance
3. bloom disappearance

## Training Settings

The following settings are reproduced from the manuscript:

- optimizer: AdamW
- initial learning rate: 6e-5
- minimum learning rate: 1e-6
- weight decay: 0.01
- batch size: 8
- total iterations: 20,000
- learning-rate schedule: 1,000-iteration linear warm-up followed by cosine annealing
- data augmentation: random rotation, random cropping, horizontal flipping, and vertical flipping
- training loss: three-class Dice loss
- validation interval: every 2,000 iterations
- checkpoint selection: highest validation mIoU
