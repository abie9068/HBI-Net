# Copyright (c) HBI-Net authors. All rights reserved.
"""HBI-Net change-detector entry point."""

from opencd.models.change_detectors.siamencoder_decoder import (
    SiamEncoderDecoder,
)
from opencd.registry import MODELS


@MODELS.register_module()
class HBINet(SiamEncoderDecoder):
    """HBI-Net change detector built on Open-CD's SiamEncoderDecoder.

    The backbone, HBI fusion neck, SC-Decoder, loss, and runtime settings are
    supplied by ``configs/hbi_net_abcd.py``. The inherited implementation
    provides training, prediction, and tensor-forward interfaces.
    """

