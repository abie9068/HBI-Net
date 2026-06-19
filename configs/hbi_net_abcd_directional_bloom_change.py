model = dict(
    name="HBI-Net",
    task="directional algal bloom change detection",
    dataset="ABCD",
    input_bands=[
        "B2", "B3", "B4", "B5", "B6",
        "B7", "B8", "B8A", "B11", "B12",
    ],
    input_rule_products=False,
    classes=["unchanged", "bloom appearance", "bloom disappearance"],
    components=[
        "shared Swin-T backbone",
        "HBI Block",
        "signed-difference fusion neck",
        "SC-Decoder",
    ],
)

training = dict(
    framework=["PyTorch", "MMEngine", "MMSegmentation", "Open-CD"],
    hardware="single NVIDIA GeForce RTX 4090 GPU",
    patch_size=(256, 256),
    optimizer=dict(type="AdamW", lr=6e-5, min_lr=1e-6, weight_decay=0.01),
    batch_size=8,
    total_iterations=20000,
    lr_schedule=dict(
        warmup="linear",
        warmup_iterations=1000,
        policy="cosine annealing",
    ),
    data_augmentation=[
        "random rotation",
        "random cropping",
        "horizontal flipping",
        "vertical flipping",
    ],
    loss="three-class Dice loss",
    validation_interval=2000,
    checkpoint_selection="highest validation mIoU",
)
