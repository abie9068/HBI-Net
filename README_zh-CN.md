# HBI-Net

[English](README.md) | **简体中文**

以下论文的官方实现：

> **HBI-Net：用于多光谱影像定向藻华变化检测的分层双时相交互网络**  
> *HBI-Net: Hierarchical Bitemporal Interaction Network for Directional Algal Bloom Change Detection in Multispectral Imagery*

## 项目概述

HBI-Net 将藻华监测建模为一个**定向变化检测**问题。给定两幅配准后的 Sentinel-2 多光谱图像块（`T1` 和 `T2`），模型输出像素级三分类结果：

- `unchanged`：未变化；
- `bloom appearance`：藻华出现（水体 → 藻华）；
- `bloom disappearance`：藻华消退（藻华 → 水体）。

HBI-Net 基于 [Open-CD](https://github.com/likyoo/open-cd)（MMEngine / MMSegmentation）构建，整体流程如下：

```text
T1 / T2 多光谱输入
        ↓
共享 Swin-T backbone
        ↓
HBI Blocks + signed-difference fusion
        ↓
SC-Decoder
        ↓
三分类定向变化图
```

本文档提供基于本仓库所用 ABCD 数据包的完整操作流程，包括环境配置、数据准备、模型训练和测试评估。

## 目录

1. [仓库结构](#1-仓库结构)
2. [论文模块与代码对应关系](#2-论文模块与代码对应关系)
3. [环境配置](#3-环境配置)
4. [数据集](#4-数据集)
5. [模型训练](#5-模型训练)
6. [测试与评估](#6-测试与评估)
7. [端到端快速开始](#7-端到端快速开始)
8. [使用自定义数据](#8-使用自定义数据)
9. [常见问题](#9-常见问题)
10. [引用](#10-引用)

## 1. 仓库结构

```text
HBI-Net/
├── configs/
│   └── hbi_net_abcd.py            # 模型、数据和训练配置
├── hbi_net/
│   ├── __init__.py                # 自定义模块注册
│   ├── hbi_neck.py                # HBIBlock 和 HBIFusionNeck
│   ├── sc_decoder.py              # SCDecoderHead
│   ├── masked_classwise_dice_loss.py # 有效像素掩膜与逐类等权 Dice loss
│   └── abcd_dataset.py            # ABCD Dataset 与 10-band .npy loader
├── tools/
│   ├── install.sh                 # 环境安装脚本
│   ├── train.py                   # 训练入口
│   └── test.py                    # 测试入口
├── docs/
│   └── method_summary.md
├── requirements.txt
├── README.md
└── README_zh-CN.md
```

导入 `hbi_net` 时，自定义组件会注册到 Open-CD。Config 中使用以下名称调用这些组件：

- `HBIFusionNeck`；
- `SCDecoderHead`；
- `MaskedClasswiseDiceLoss`；
- `ABCDDirectionalBloomDataset`；
- `MultiImgLoadMSImageFromNpy`。

## 2. 论文模块与代码对应关系

| 论文模块 | 代码实现 | 说明 |
|---|---|---|
| 共享 Swin-T backbone | [`configs/hbi_net_abcd.py`](configs/hbi_net_abcd.py) 中的 `mmseg.SwinTransformer` | `T1` 与 `T2` 共享参数，`in_channels=10` |
| 多光谱输入嵌入 | Swin Transformer patch embedding | 4 × 4 非重叠 patch → 96 维 token |
| HBI Block：context branch φ | [`hbi_net/hbi_neck.py`](hbi_net/hbi_neck.py) 中的 `HBIBlock` | 四个 grouped 3 × 3 dilated convolution（`d=1,2,3,4`，`groups=C`） |
| HBI Block：signed-difference gate ψ | `HBIBlock.gate` | 对 `F2 − F1` 使用 depth-wise 3 × 3 convolution + BN，随后使用 `sigmoid` |
| 反对称残差更新 | `HBIBlock.forward` | `F1' = F1 − αΔ`，`F2' = F2 + αΔ`；可学习的 `α` 初始化为 0.1 |
| Signed-difference fusion | `HBIFusionNeck.forward` | 在每个尺度计算 `D_i = F2'_i − F1'_i` |
| SC-Decoder | [`hbi_net/sc_decoder.py`](hbi_net/sc_decoder.py) 中的 `SCDecoderHead` | 在 UPer 路径中加入 SE channel 和 CBAM spatial recalibration |
| 三分类 Dice loss | [`hbi_net/masked_classwise_dice_loss.py`](hbi_net/masked_classwise_dice_loss.py) 中的 `MaskedClasswiseDiceLoss` | 仅使用有效像素，三个类别等权 |
| 训练方案 | `optim_wrapper`、`param_scheduler` 和 `train_cfg` | AdamW、1,000 iterations warm-up、cosine annealing、20,000 iterations |

## 3. 环境配置

### 已验证环境

| 组件 | 版本 |
|---|---|
| 操作系统 | Ubuntu 22.04 |
| GPU | 单张 NVIDIA RTX 4090 |
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

### 推荐安装方式

```bash
conda create -n hbinet python=3.10 -y
conda activate hbinet
bash tools/install.sh
```

<details>
<summary>展开查看手动安装命令</summary>

```bash
# 1) PyTorch（根据 CUDA 版本调整 index URL）
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) OpenMMLab 核心组件
pip install "numpy<2.1" "mmengine==0.10.7"
pip install "mmcv==2.2.0" \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
pip install "mmsegmentation==1.2.2" "mmdet==3.3.0" \
    "mmpretrain>=1.0.0" ftfy regex

# 3) 从源代码安装 Open-CD
git clone https://github.com/likyoo/open-cd.git
pip install -e open-cd --no-deps
```

</details>

> **版本上限说明：** `mmcv==2.2.0` 比 `mmseg 1.2.2`、`mmdet 3.3.0` 和 `open-cd 1.1.0` 中设置的版本上限高一个 minor version。HBI-Net 使用的相关 API 与该版本兼容，因此 `tools/install.sh` 会将相应的 `MMCV_MAX` / `mmcv_maximum_version` 常量放宽至 `2.3.0`，除此之外不会修改其他第三方代码。

验证安装结果：

```bash
python -c "import torch, mmcv, mmseg, mmdet, opencd; from mmcv.ops import nms; \
print(torch.__version__, mmcv.__version__, mmseg.__version__, \
mmdet.__version__, opencd.__version__)"
```

预期输出：

```text
2.4.1 2.2.0 1.2.2 3.3.0 1.1.0
```

## 4. 数据集

本仓库使用的 ABCD 数据包 `ABCD-ms10` 可从 [ABCD Dataset release](https://github.com/abie9068/ABCD-Dataset/releases/tag/ABCD-Dataset) 获取。

### 下载数据

```bash
mkdir -p datasets
cd datasets
curl -L -o ABCD-ms10.zip \
  https://github.com/abie9068/ABCD-Dataset/releases/download/ABCD-Dataset/ABCD-ms10.zip
unzip -q ABCD-ms10.zip
cd ..
```

### 目录结构

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

### 数据格式

| 内容 | 格式 | 说明 |
|---|---|---|
| `A/`、`B/` | `.npy`，`(256, 256, 10)`，`int16` | 配准后的 `T1` / `T2` Sentinel-2 L2A surface-reflectance patches |
| Bands | 10 channels | B2、B3、B4、B5、B6、B7、B8、B8A、B11、B12 |
| `label/` | `.png`，`(256, 256)`，`uint8` | `0`：未变化；`1`：藻华出现；`2`：藻华消退；`255`：忽略像素 |

Config 默认读取仓库根目录下的 `datasets/ABCD-ms10`。若数据位于其他路径，可使用：

```bash
python tools/train.py configs/hbi_net_abcd.py \
    --cfg-options data_root=/path/to/ABCD-ms10
```

<details>
<summary>展开查看如何重新计算各波段归一化统计量</summary>

Config 中的 `band_mean` 和 `band_std` 根据 `ABCD-ms10` training split 计算。使用其他数据划分时，可通过以下脚本重新计算：

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

## 5. 模型训练

请在仓库根目录运行：

```bash
python tools/train.py configs/hbi_net_abcd.py \
    --work-dir work_dirs/hbi_net_abcd
```

### 训练配置

| 设置 | 取值 |
|---|---|
| Optimizer | AdamW |
| 初始 learning rate | `6e-5` |
| 最小 learning rate | `1e-6` |
| Weight decay | `0.01` |
| Batch size | `8` |
| 总 iterations | `20,000` |
| Learning-rate schedule | 1,000-iteration linear warm-up，随后 cosine annealing |
| Data augmentation | 随机旋转、从每个 `256 × 256` patch 随机裁剪 `224 × 224`、水平/垂直翻转 |
| Validation interval | 每 2,000 iterations 验证一次 |
| Checkpoint selection | 选择 validation `mIoU` 最高的模型 |
| Loss | 带有效像素掩膜的逐类三分类 Dice loss；三类等权；排除标签 `255` |

最优 checkpoint 保存为 `work_dirs/hbi_net_abcd/best_mIoU_iter_*.pth`。

在单张 RTX 4090 上，一次完整训练大约需要 50 分钟（约 0.15 s/iteration），batch size 为 8 时显存占用低于 5 GB。

常用参数覆盖示例：

```bash
# 短流程 smoke test
python tools/train.py configs/hbi_net_abcd.py \
    --cfg-options train_cfg.max_iters=2000 train_cfg.val_interval=500

# Mixed precision 和自定义数据路径
python tools/train.py configs/hbi_net_abcd.py --amp \
    --cfg-options data_root=/data/ABCD-ms10
```

## 6. 测试与评估

```bash
python tools/test.py configs/hbi_net_abcd.py \
    work_dirs/hbi_net_abcd/best_mIoU_iter_XXXXX.pth
```

测试程序输出以下指标：

- `mIoU` 和各类别 IoU；
- `mFscore` 和各类别 F-score；
- `mPrecision`；
- `mRecall`；
- overall accuracy（`aAcc`）。

使用以下命令保存彩色 prediction visualizations：

```bash
python tools/test.py configs/hbi_net_abcd.py \
    work_dirs/hbi_net_abcd/best_mIoU_iter_XXXXX.pth \
    --show-dir vis/
```

## 7. 端到端快速开始

```bash
# 0) 环境
conda create -n hbinet python=3.10 -y
conda activate hbinet
git clone https://github.com/abie9068/HBI-Net.git
cd HBI-Net
bash tools/install.sh

# 1) 数据
mkdir -p datasets
cd datasets
curl -L -o ABCD-ms10.zip \
  https://github.com/abie9068/ABCD-Dataset/releases/download/ABCD-Dataset/ABCD-ms10.zip
unzip -q ABCD-ms10.zip
cd ..

# 2) 训练
python tools/train.py configs/hbi_net_abcd.py \
    --work-dir work_dirs/hbi_net_abcd

# 3) 使用最佳 checkpoint 进行测试
python tools/test.py configs/hbi_net_abcd.py \
    work_dirs/hbi_net_abcd/$(ls -t work_dirs/hbi_net_abcd | grep best_mIoU | head -1)
```

## 8. 使用自定义数据

若要训练其他三分类定向变化检测数据集：

1. 按照相同的 `train/val/test` 与 `A/B/label` 结构整理数据。
2. 将两个时相的图像保存为 10-band `.npy` 数组，将标签保存为取值 `{0,1,2,255}` 的 PNG mask。
3. 使用新的 training split 重新计算 `band_mean` 和 `band_std`。
4. 将 `data_root` 指向新的数据集位置。

若输入 bands 数量不同，需要同时修改：

- `model.backbone.in_channels`；
- `model.backbone_inchannels`；
- `band_mean` 和 `band_std` 的长度。

标准 RGB 变化检测数据集可以使用 Open-CD 的 `MultiImgLoadImageFromFile`，替代 `MultiImgLoadMSImageFromNpy`。

## 9. 常见问题

| 问题 | 建议处理方式 |
|---|---|
| `MMCV==2.2.0 is used but incompatible` | 运行 `tools/install.sh`，或安装与 PyTorch、CUDA 匹配的 `mmcv<2.2.0` |
| `No module named 'hbi_net'` | 从仓库根目录运行命令，并确保根目录已加入 `PYTHONPATH` |
| `No module named 'ftfy'` | 运行 `pip install ftfy regex` |
| 单 GPU 出现 `SyncBN` 或 distributed-training 错误 | 单 GPU 运行时保留 `norm_cfg=dict(type='BN')` |
| CUDA out of memory | 通过 `--cfg-options` 减小 `train_dataloader.batch_size` |
| GitHub 下载缓慢或中断 | 使用 `curl -L -C -` 继续未完成的下载 |

## 10. 引用

如果本项目对你的研究有帮助，请引用 HBI-Net 手稿及其所基于的开源框架。HBI-Net 的完整 BibTeX 信息将在论文正式发表后补充。

- **Open-CD** — Li *et al.*, “Open-CD: A Comprehensive Toolbox for Change Detection,” *ACM MM*, 2025。
- **MMSegmentation** — MMSegmentation Contributors, OpenMMLab, 2020。
- **Swin Transformer** — Liu *et al.*, *ICCV*, 2021。
