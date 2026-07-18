# HBI-Net Method Summary

HBI-Net performs **directional** algal-bloom change detection on the ABCD
dataset: from a co-registered Sentinel-2 pair `(T1, T2)` it predicts a
pixel-level three-class map (unchanged / bloom appearance / bloom
disappearance).

This document summarizes the method. The runnable implementation lives in
[`../configs/hbi_net_abcd.py`](../configs/hbi_net_abcd.py) and the
[`../hbi_net/`](../hbi_net) package; see the top-level
[README](../README.md) for the full reproduction workflow and the
manuscript ↔ code map.

## Input

Paired Sentinel-2 10-band L2A surface-reflectance patches:

- `T1`: earlier-time patch, shape `10 × 256 × 256`
- `T2`: later-time patch, shape `10 × 256 × 256`

Bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12. FAI, threshold maps, NDCI,
SCL, AOT, WVP, water masks and other rule-derived products are used only for
label generation / quality control and are **not** model inputs.

## Output

Pixel-level three-class prediction: unchanged, bloom appearance, bloom
disappearance.

## Components

| Component | Role | Code |
|---|---|---|
| Shared Swin-T backbone | multi-scale spatial-spectral features from both dates (shared weights) | `mmseg.SwinTransformer`, `in_channels=10` |
| HBI Block | direction-aware bitemporal interaction via signed-difference gating | `hbi_net/hbi_neck.py::HBIBlock` |
| Signed-difference fusion neck | `D_i = F2'_i − F1'_i`, preserving the T1→T2 order | `hbi_net/hbi_neck.py::HBIFusionNeck` |
| SC-Decoder | channel + spatial recalibration on a UPer decoding path | `hbi_net/sc_decoder.py::SCDecoderHead` |

### HBI Block (per scale)

```
Z   = concat(F1, F2)                          # 2C channels
R   = refine(axial(mix( [conv_d(Z) for d in (1,2,3,4)] )))   # context branch phi
G   = sigmoid( BN(dwconv3x3(F2 - F1)) )        # signed-difference gate psi
delta = R * G
F1' = F1 - a*delta ,  F2' = F2 + a*delta       # anti-symmetric update, a init 0.1
```

The opposite-sign update keeps the temporal order instead of collapsing it into
an unsigned magnitude, which is what lets the model separate bloom *appearance*
from *disappearance*.

## Training Settings

- optimizer: AdamW
- initial learning rate: 6e-5
- minimum learning rate: 1e-6
- weight decay: 0.01
- batch size: 8
- total iterations: 20,000
- learning-rate schedule: 1,000-iteration linear warm-up followed by cosine annealing
- data augmentation: random rotation, random 224 × 224 crops from the 256 × 256 source patches, horizontal flipping, and vertical flipping
- training loss: masked class-wise three-class Dice loss (equal class weights; label 255 excluded)
- validation interval: every 2,000 iterations
- checkpoint selection: highest validation mIoU
