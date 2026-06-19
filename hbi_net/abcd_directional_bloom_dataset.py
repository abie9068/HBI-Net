class ABCDDirectionalBloomDataset:
    """Reference interface for the ABCD directional algal bloom dataset.

    This class documents the expected dataset interface.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Use the project-specific ABCD dataset loader for execution."
        )

    def __getitem__(self, index):
        """Return one paired sample."""
        raise NotImplementedError(
            "Use the project-specific sample loading logic for execution."
        )

    def __len__(self):
        """Return dataset length."""
        raise NotImplementedError(
            "Use the project-specific dataset length logic for execution."
        )
