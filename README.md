# HBI-Net

**English** | [简体中文](README_zh-CN.md)

Official implementation of:

> **HBI-Net: Hierarchical Bitemporal Interaction Network for Directional Algal Bloom Change Detection in Multispectral Imagery**

## Overview

HBI-Net formulates algal-bloom monitoring as a **directional change-detection** problem. Given two co-registered Sentinel-2 multispectral patches (`T1` and `T2`), it predicts a pixel-level three-class map:

- `unchanged`;
- `bloom appearance` (water → bloom);
- `bloom disappearance` (bloom → water).

HBI-Net is built on [Open-CD](https://github.com/likyoo/open-cd) (MMEngine / MMSegmentation) and follows this workflow:

```text
T1 / T2 multispectral inputs
           ↓
Shared Swin-T backbone
           ↓
HBI Blocks + signed-difference fusion
           ↓
SC-Decoder
           ↓
Three-class directional change map
```

This README provides an end-to-end workflow for environment setup, data preparation, training, and evaluation with the ABCD data package used by this repository.

## Contents

1. [Repository layout](#1-repository-layout)
2. [Manuscript-to-code map](#2-manuscript-to-code-map)
3. [Environment setup](#3-environment-setup)
4. [Dataset](#4-dataset)
5. [Training](#5-training)
6. [Testing and evaluation](#6-testing-and-evaluation)
7. [End-to-end quick start](#7-end-to-end-quick-start)
8. [Using your own data](#8-using-your-own-data)
9. [Troubleshooting](#9-troubleshooting)
10. [Citation](#10-citation)

## 1. Repository layout

```text
HBI-Net/
├── configs/
│   └── hbi_net_abcd.py            # model, data, and training configuration
├── hbi_net/
│   ├── __init__.py                # custom-module registration
│   ├── hbi_neck.py                # HBIBlock and HBIFusionNeck
│   ├── sc_decoder.py              # SCDecoderHead
│   ├── masked_classwise_dice_loss.py # masked, equal-weight class-wise Dice
│   └── abcd_dataset.py            # ABCD dataset and 10-band .npy loader
├── tools/
│   ├── install.sh                 # environment setup
│   ├── train.py                   # training entry point
│   └── test.py                    # evaluation entry point
├── docs/
│   └── method_summary.md
├── requirements.txt
├── README.md
└── README_zh-CN.md
```

Importing `hbi_net` registers the custom components in Open-CD. The configuration refers to them as:

- `HBIFusionNeck`;
- `SCDecoderHead`;
- `MaskedClasswiseDiceLoss`;
- `ABCDDirectionalBloomDataset`;
- `MultiImgLoadMSImageFromNpy`.

## 2. Manuscript-to-code map

| Manuscript component | Implementation | Notes |
|---|---|---|
| Shared Swin-T backbone | `mmseg.SwinTransformer` in [`configs/hbi_net_abcd.py`](configs/hbi_net_abcd.py) | Shared parameters for `T1` and `T2`; `in_channels=10` |
| Multispectral input embedding | Swin Transformer patch embedding | 4 × 4 non-overlapping patches → 96-dimensional tokens |
| HBI Block: context branch φ | `HBIBlock` in [`hbi_net/hbi_neck.py`](hbi_net/hbi_neck.py) | Four grouped 3 × 3 dilated convolutions (`d=1,2,3,4`, `groups=C`) |
| HBI Block: signed-difference gate ψ | `HBIBlock.gate` | Depth-wise 3 × 3 convolution + BN on `F2 − F1`, followed by `sigmoid` |
| Anti-symmetric residual update | `HBIBlock.forward` | `F1' = F1 − αΔ`, `F2' = F2 + αΔ`; learnable `α` initialized to 0.1 |
| Signed-difference fusion | `HBIFusionNeck.forward` | `D_i = F2'_i − F1'_i` at each scale |
| SC-Decoder | `SCDecoderHead` in [`hbi_net/sc_decoder.py`](hbi_net/sc_decoder.py) | UPer path with SE channel and CBAM spatial recalibration |
| Three-class Dice loss | `MaskedClasswiseDiceLoss` in [`hbi_net/masked_classwise_dice_loss.py`](hbi_net/masked_classwise_dice_loss.py) | Valid pixels only; equal weighting of the three classes |
| Training protocol | `optim_wrapper`, `param_scheduler`, and `train_cfg` | AdamW, 1,000-iteration warm-up, cosine annealing, 20,000 iterations |

## 3. Environment setup

### Verified environment

| Component | Version |
|---|---|
| Operating system | Ubuntu 22.04 |
| GPU | Single NVIDIA RTX 4090 |
| CUDA | 12.1 |
| Python | 3.10 |
| PyTorch | 2.4.1 |
| TorchVision | 0.19.1 |
| NumPy | 2.0.1 |
| MMEngine | 0.10.7 |
| MMCV | 2.2.0 |
| MMSegmentation | 1.2.2 |
| MMDetection | 3.3.0 |
| Open-CD | 1.1.0 |

### Recommended installation

```bash
conda create -n hbinet python=3.10 -y
conda activate hbinet
bash tools/install.sh
```

<details>
<summary>Manual installation commands</summary>

```bash
# 1) PyTorch (match the index URL to your CUDA version)
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) OpenMMLab core
pip install "numpy<2.1" "mmengine==0.10.7"
pip install "mmcv==2.2.0" \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
pip install "mmsegmentation==1.2.2" "mmdet==3.3.0" \
    "mmpretrain>=1.0.0" ftfy regex

# 3) Open-CD from source
git clone https://github.com/likyoo/open-cd.git
pip install -e open-cd --no-deps
```

</details>

> **Version-cap note:** `mmcv==2.2.0` is one minor version above the upper bound hard-coded in `mmseg 1.2.2`, `mmdet 3.3.0`, and `open-cd 1.1.0`. The APIs used by HBI-Net are compatible with this version. Therefore, `tools/install.sh` relaxes the corresponding `MMCV_MAX` / `mmcv_maximum_version` constants to `2.3.0`. No other third-party code is modified.

Verify the installation:

```bash
python -c "import torch, mmcv, mmseg, mmdet, opencd; from mmcv.ops import nms; \
print(torch.__version__, mmcv.__version__, mmseg.__version__, \
mmdet.__version__, opencd.__version__)"
```

Expected output:

```text
2.4.1 2.2.0 1.2.2 3.3.0 1.1.0
```

## 4. Dataset

The ABCD data package used by this repository, `ABCD-ms10`, is available from the [ABCD Dataset release](https://github.com/abie9068/ABCD-Dataset/releases/tag/ABCD-Dataset).

### Download

```bash
mkdir -p datasets
cd datasets
curl -L -o ABCD-ms10.zip \
  https://github.com/abie9068/ABCD-Dataset/releases/download/ABCD-Dataset/ABCD-ms10.zip
unzip -q ABCD-ms10.zip
cd ..
```

### Expected layout

```text
datasets/ABCD-ms10/
├── train/
│   ├── A/*.npy
│   ├── B/*.npy
│   └── label/*.png               # 60 patches
├── val/
│   ├── A/*.npy
│   ├── B/*.npy
│   └── label/*.png               # 20 patches
└── test/
    ├── A/*.npy
    ├── B/*.npy
    └── label/*.png               # 20 patches
```

### Data specification

| Item | Format | Description |
|---|---|---|
| `A/`, `B/` | `.npy`, `(256, 256, 10)`, `int16` | Co-registered `T1` / `T2` Sentinel-2 L2A surface-reflectance patches |
| Bands | 10 channels | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |
| `label/` | `.png`, `(256, 256)`, `uint8` | `0`: unchanged; `1`: bloom appearance; `2`: bloom disappearance; `255`: ignored |

The default data location is `datasets/ABCD-ms10`, relative to the repository root. To use another location:

```bash
python tools/train.py configs/hbi_net_abcd.py \
    --cfg-options data_root=/path/to/ABCD-ms10
```

<details>
<summary>Recompute per-band normalization statistics</summary>

The config contains `band_mean` and `band_std` values computed from the `ABCD-ms10` training split. Recompute them when using a different split:

```bash
python - <<'PY'
import glob
import numpy as np

files = glob.glob('datasets/ABCD-ms10/train/A/*.npy') + \
        glob.glob('datasets/ABCD-ms10/train/B/*.npy')
pixels = np.stack([
    np.load(path).astype(np.float64).reshape(-1, 10)
    for path in files
]).reshape(-1, 10)

print('mean =', [round(value, 3) for value in pixels.mean(0)])
print('std  =', [round(value, 3) for value in pixels.std(0)])
PY
```

</details>

## 5. Training

Run training from the repository root:

```bash
python tools/train.py configs/hbi_net_abcd.py \
    --work-dir work_dirs/hbi_net_abcd
```

### Training configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Initial learning rate | `6e-5` |
| Minimum learning rate | `1e-6` |
| Weight decay | `0.01` |
| Batch size | `8` |
| Total iterations | `20,000` |
| Learning-rate schedule | 1,000-iteration linear warm-up followed by cosine annealing |
| Data augmentation | Random rotation, random `224 × 224` crop from each `256 × 256` patch, horizontal/vertical flipping |
| Validation interval | Every 2,000 iterations |
| Checkpoint selection | Highest validation `mIoU` |
| Loss | Masked class-wise three-class Dice loss; equal class weights; label `255` excluded |

The best checkpoint is saved as `work_dirs/hbi_net_abcd/best_mIoU_iter_*.pth`.

On a single RTX 4090, a full run takes approximately 50 minutes (~0.15 s/iteration) and uses less than 5 GB of VRAM at batch size 8.

Useful overrides:

```bash
# Short smoke test
python tools/train.py configs/hbi_net_abcd.py \
    --cfg-options train_cfg.max_iters=2000 train_cfg.val_interval=500

# Mixed precision and a custom data location
python tools/train.py configs/hbi_net_abcd.py --amp \
    --cfg-options data_root=/data/ABCD-ms10
```

## 6. Testing and evaluation

```bash
python tools/test.py configs/hbi_net_abcd.py \
    work_dirs/hbi_net_abcd/best_mIoU_iter_XXXXX.pth
```

The evaluation reports:

- `mIoU` and class-wise IoU;
- `mFscore` and class-wise F-score;
- `mPrecision`;
- `mRecall`;
- overall accuracy (`aAcc`).

Save color-coded prediction visualizations with:

```bash
python tools/test.py configs/hbi_net_abcd.py \
    work_dirs/hbi_net_abcd/best_mIoU_iter_XXXXX.pth \
    --show-dir vis/
```

## 7. End-to-end quick start

```bash
# 0) Environment
conda create -n hbinet python=3.10 -y
conda activate hbinet
git clone https://github.com/abie9068/HBI-Net.git
cd HBI-Net
bash tools/install.sh

# 1) Data
mkdir -p datasets
cd datasets
curl -L -o ABCD-ms10.zip \
  https://github.com/abie9068/ABCD-Dataset/releases/download/ABCD-Dataset/ABCD-ms10.zip
unzip -q ABCD-ms10.zip
cd ..

# 2) Training
python tools/train.py configs/hbi_net_abcd.py \
    --work-dir work_dirs/hbi_net_abcd

# 3) Testing with the best checkpoint
python tools/test.py configs/hbi_net_abcd.py \
    work_dirs/hbi_net_abcd/$(ls -t work_dirs/hbi_net_abcd | grep best_mIoU | head -1)
```

## 8. Using your own data

To train on another three-class directional change-detection dataset:

1. Arrange the data using the same `train/val/test` and `A/B/label` structure.
2. Store each temporal image as a 10-band `.npy` array and each label as a `{0,1,2,255}` PNG mask.
3. Recompute `band_mean` and `band_std` for the new training split.
4. Set `data_root` to the new dataset location.

For a different number of input bands, update:

- `model.backbone.in_channels`;
- `model.backbone_inchannels`;
- the lengths of `band_mean` and `band_std`.

Standard RGB change-detection datasets can use Open-CD's `MultiImgLoadImageFromFile` instead of `MultiImgLoadMSImageFromNpy`.

## 9. Troubleshooting

| Symptom | Suggested action |
|---|---|
| `MMCV==2.2.0 is used but incompatible` | Run `tools/install.sh`, or install a compatible `mmcv<2.2.0` build for your PyTorch and CUDA versions |
| `No module named 'hbi_net'` | Run commands from the repository root and ensure the root is included in `PYTHONPATH` |
| `No module named 'ftfy'` | Run `pip install ftfy regex` |
| `SyncBN` or distributed-training error on one GPU | Keep `norm_cfg=dict(type='BN')` for single-GPU execution |
| CUDA out of memory | Reduce `train_dataloader.batch_size` through `--cfg-options` |
| Slow or interrupted GitHub download | Retry with `curl -L -C -` to resume the partial download |

## 10. Citation

If you find this work useful, please cite the manuscript and the frameworks on which it is built. The complete BibTeX entry for HBI-Net will be added after publication.

- **Open-CD** — Li *et al.*, “Open-CD: A Comprehensive Toolbox for Change Detection,” *ACM MM*, 2025.
- **MMSegmentation** — MMSegmentation Contributors, OpenMMLab, 2020.
- **Swin Transformer** — Liu *et al.*, *ICCV*, 2021.
