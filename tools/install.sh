#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup for HBI-Net (Open-CD backend).
#
# Sets up the HBI-Net software environment for training and evaluation with
# the currently released 100-patch ABCD-ms10 preview subset. Verified on a
# single NVIDIA RTX 4090 with CUDA 12.1 and Python 3.10.
#
# Usage:
#   conda create -n hbinet python=3.10 -y && conda activate hbinet
#   bash tools/install.sh
# ---------------------------------------------------------------------------
set -e

# 1. PyTorch (CUDA 12.1 build). Change the index-url to match your CUDA.
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. OpenMMLab core. mmcv is pulled from the wheel index that matches
#    torch2.4/cu121 (only mmcv==2.2.0 has a prebuilt wheel there).
pip install "numpy<2.1" "mmengine==0.10.7"
pip install "mmcv==2.2.0" \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
pip install "mmsegmentation==1.2.2" "mmdet==3.3.0" "mmpretrain>=1.0.0" ftfy regex

# 3. Open-CD (installed from source, editable).
if [ ! -d open-cd ]; then
    git clone https://github.com/likyoo/open-cd.git
fi
pip install -e open-cd --no-deps

# 4. mmcv 2.2.0 is one minor version above the caps hard-coded in
#    mmseg 1.2.2 / mmdet 3.3.0 / open-cd 1.1.0. The 2.1 -> 2.2 change is
#    API-compatible for everything HBI-Net uses, so we relax the caps.
python - <<'PY'
import re, mmseg, mmdet, opencd, os
def bump(path, patterns):
    with open(path) as f:
        src = f.read()
    for old, new in patterns:
        src = src.replace(old, new)
    with open(path, 'w') as f:
        f.write(src)
    print('configured dependency version bound:', path)

mmseg_init = os.path.join(os.path.dirname(mmseg.__file__), '__init__.py')
mmdet_init = os.path.join(os.path.dirname(mmdet.__file__), '__init__.py')
opencd_init = os.path.join(os.path.dirname(opencd.__file__), '__init__.py')
bump(mmseg_init, [("MMCV_MAX = '2.2.0'", "MMCV_MAX = '2.3.0'")])
bump(mmdet_init, [("mmcv_maximum_version = '2.2.0'",
                   "mmcv_maximum_version = '2.3.0'")])
bump(opencd_init, [("mmcv_maximum_version = '2.2.0'",
                    "mmcv_maximum_version = '2.3.0'")])
PY

# 5. Sanity check.
python -c "import torch, mmcv, mmengine, mmseg, mmdet, opencd; \
print('torch', torch.__version__, '| cuda', torch.version.cuda); \
print('mmcv', mmcv.__version__, '| mmseg', mmseg.__version__, \
'| mmdet', mmdet.__version__, '| opencd', opencd.__version__); \
from mmcv.ops import nms; print('mmcv CUDA ops OK')"

echo
echo "Environment ready. Next: download the dataset (see README.md) and run"
echo "  python tools/train.py configs/hbi_net_abcd.py --work-dir work_dirs/hbi_net_abcd"
