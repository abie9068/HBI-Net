# HBI-Net Method Summary

HBI-Net is designed for directional algal bloom change detection on the ABCD dataset.

This document summarizes the method interface and manuscript-level settings.

## Input

The model uses paired Sentinel-2 10-band surface-reflectance patches:

- `T1`: earlier-time patch, shape `10 x 256 x 256`
- `T2`: later-time patch, shape `10 x 256 x 256`

FAI, threshold maps, NDCI, SCL, AOT, WVP, water masks and other rule-derived products are not used as model inputs.

## Output

The output is a pixel-level three-class prediction:

- unchanged
- bloom appearance
- bloom disappearance

## Components

- shared Swin-T backbone
- HBI Block
- signed-difference fusion neck
- SC-Decoder

## Training Settings

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

For additional materials, please contact the corresponding author.
