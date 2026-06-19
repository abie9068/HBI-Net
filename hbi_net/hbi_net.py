class HBINet:
    """Reference interface for HBI-Net.

    The manuscript describes HBI-Net as a shared Swin-T backbone with HBI Blocks,
    a signed-difference fusion neck, and an SC-Decoder.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Use the project-specific HBI-Net implementation for execution."
        )

    def forward(self, t1, t2):
        """Predict unchanged, bloom appearance, and bloom disappearance maps."""
        raise NotImplementedError(
            "Use the project-specific forward computation for execution."
        )
