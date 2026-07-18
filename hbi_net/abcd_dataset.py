# Copyright (c) HBI-Net authors. All rights reserved.
"""ABCD directional algal-bloom change-detection dataset and 10-band loader.

The public ABCD release (``ABCD-ms10``) uses the standard Open-CD
change-detection layout::

    ABCD-ms10/
        train/  A/*.npy   B/*.npy   label/*.png
        val/    A/*.npy   B/*.npy   label/*.png
        test/   A/*.npy   B/*.npy   label/*.png

* ``A`` and ``B`` hold the co-registered bitemporal Sentinel-2 patches as
  ``(256, 256, 10)`` ``int16`` surface-reflectance arrays (bands B2, B3, B4,
  B5, B6, B7, B8, B8A, B11, B12).
* ``label`` holds ``(256, 256)`` ``uint8`` PNG masks with three directional
  classes: ``0`` unchanged, ``1`` bloom appearance, ``2`` bloom disappearance.
"""
from typing import Optional

import numpy as np

from mmcv.transforms import BaseTransform

from opencd.registry import DATASETS, TRANSFORMS
from opencd.datasets.basecddataset import _BaseCDDataset


@DATASETS.register_module()
class ABCDDirectionalBloomDataset(_BaseCDDataset):
    """ABCD directional algal-bloom change-detection dataset.

    Three directional classes (unchanged / bloom appearance / bloom
    disappearance) and the 10-band Sentinel-2 ``.npy`` / ``.png`` layout.
    """

    METAINFO = dict(
        classes=('unchanged', 'bloom appearance', 'bloom disappearance'),
        palette=[[0, 0, 0], [0, 0, 255], [255, 0, 0]])

    def __init__(self,
                 img_suffix='.npy',
                 seg_map_suffix='.png',
                 format_seg_map=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)


@TRANSFORMS.register_module()
class MultiImgLoadMSImageFromNpy(BaseTransform):
    """Load a bitemporal pair of multispectral ``.npy`` patches.

    Mirrors the contract of Open-CD's ``MultiImgLoadImageFromFile`` but reads
    ``.npy`` arrays (H, W, C) instead of decoding 8-bit images, so that the
    full 10-band Sentinel-2 reflectance is preserved for both dates.

    Required Keys:
        - img_path (list[str]): ``[path_to_A, path_to_B]``

    Modified Keys:
        - img (list[np.ndarray]): ``[A, B]`` each ``(H, W, C)`` float32
        - img_shape
        - ori_shape
    """

    def __init__(self, to_float32: bool = True, ignore_empty: bool = False):
        self.to_float32 = to_float32
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> Optional[dict]:
        filenames = results['img_path']
        imgs = []
        try:
            for filename in filenames:
                img = np.load(filename)
                if self.to_float32:
                    img = img.astype(np.float32)
                if img.ndim == 2:
                    img = img[..., None]
                imgs.append(img)
        except Exception as e:
            if self.ignore_empty:
                return None
            raise e

        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f'ignore_empty={self.ignore_empty})')
