# HBI-Net on the ABCD directional algal-bloom change-detection dataset.
#
# Self-contained MMEngine / Open-CD config. HBI-Net is assembled from three
# registered components:
#
#   shared Swin-T backbone  ->  HBIFusionNeck (HBI Block + signed-difference
#   fusion)  ->  SCDecoderHead (Spatial-Channel Decoder)  ->  3-class Dice loss
#
# The ``custom_imports`` line registers the HBI-Net modules; make sure the repo
# root is on PYTHONPATH (the provided tools/train.py and tools/test.py already
# ``import hbi_net``).

custom_imports = dict(imports=['hbi_net'], allow_failed_imports=False)

# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='CDLocalVisBackend')]
visualizer = dict(
    type='CDLocalVisualizer', vis_backends=vis_backends,
    name='visualizer', alpha=1.0)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=2000,
        save_best='mIoU', rule='greater', max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1))

# ---------------------------------------------------------------------------
# data preprocessing / normalization
# ---------------------------------------------------------------------------
# 10-band Sentinel-2 surface-reflectance statistics computed on the ABCD-ms10
# training split (bands B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12); repeated for T1/T2.
band_mean = [969.444, 1287.807, 997.967, 1159.794, 829.206,
             806.795, 729.076, 606.654, 174.110, 120.915]
band_std = [191.344, 175.351, 252.845, 301.566, 535.782,
            547.187, 522.726, 508.973, 148.554, 107.925]
# Training uses a genuine random crop from each 256 x 256 source patch.
# Evaluation keeps the full patch; ``test_cfg`` pads only when required.
train_crop_size = (224, 224)

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=band_mean * 2,
    std=band_std * 2,
    bgr_to_rgb=False,           # multispectral: keep native band order
    size=train_crop_size,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

# ---------------------------------------------------------------------------
# model: HBI-Net
# ---------------------------------------------------------------------------
norm_cfg = dict(type='BN', requires_grad=True)   # single-GPU friendly

model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone_inchannels=10,     # 10 Sentinel-2 bands per date
    backbone=dict(
        type='mmseg.SwinTransformer',
        in_channels=10,
        embed_dims=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False),
    neck=dict(
        type='HBIFusionNeck',
        in_channels=[96, 192, 384, 768],
        dilations=(1, 2, 3, 4),
        alpha_init=0.1,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='SCDecoderHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        se_reduction=16,
        spatial_kernel=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='MaskedClasswiseDiceLoss',
            eps=1e-3,
            ignore_index=255,
            reduction='mean',
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
dataset_type = 'ABCDDirectionalBloomDataset'
data_root = 'datasets/ABCD-ms10'

train_pipeline = [
    dict(type='MultiImgLoadMSImageFromNpy'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5,
         degree=(-20, 20)),
    dict(type='MultiImgRandomCrop', crop_size=train_crop_size,
         cat_max_ratio=0.9),
    dict(type='MultiImgPackSegInputs'),
]
test_pipeline = [
    dict(type='MultiImgLoadMSImageFromNpy'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='train/A',
            img_path_to='train/B',
            seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='val/A',
            img_path_to='val/B',
            seg_map_path='val/label'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='test/A',
            img_path_to='test/B',
            seg_map_path='test/label'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='mmseg.IoUMetric',
    iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

# ---------------------------------------------------------------------------
# optimization schedule  (manuscript settings)
# ---------------------------------------------------------------------------
optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))

# 1,000-iter linear warm-up + cosine annealing to min-lr 1e-6, total 20,000
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=1e-6, by_epoch=False,
         begin=1000, end=20000),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
