weight = '/home/mbassier/code/Scan-to-BIM-CVPR-2024/data/t1/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 48380007
save_path = '/home/mbassier/code/Scan-to-BIM-CVPR-2024/data/t1'
num_worker = 3 #workers are divided per gpu (minimum three in our case)
batch_size = 6 #must be divisible by  nr of gpu's 
batch_size_val = None #this doesn't seem to matter
batch_size_test = None #this doesn't seem to matter
epoch = 800
eval_epoch = 100
sync_bn = False
enable_amp = True
empty_cache = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0006)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='DefaultSegmentorV2',
    num_classes=6,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=6,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(64, 64, 64, 64, 64),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(64, 64, 64, 64),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions='ThreeDom'),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'KULDataset'
data_root = '/home/mbassier/code/Scan-to-BIM-CVPR-2024/data/t1'

data = dict(
    num_classes=6,
    ignore_index=5,
    names=['floors', 'ceiling', 'wall', 'column', 'door','unassigned'],
    labels=[0, 1, 2, 3, 4, 255],
    train=dict(
        type='KULDataset',
        split='train',
        data_root=data_root,
        transform=[
            # dict(type='CenterShift', apply_z=True),
            # dict(
            #     type='RandomDropout',
            #     dropout_ratio=0.2,
            #     dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            # dict(
            #     type='RandomRotate',
            #     angle=[-0.015625, 0.015625],
            #     axis='x',
            #     p=0.5),
            # dict(
            #     type='RandomRotate',
            #     angle=[-0.015625, 0.015625],
            #     axis='y',
            #     p=0.5),
            # dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            # dict(
            #     type='ElasticDistortion',
            #     distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type='GridSample',
                grid_size=0.04,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=102400, mode='random'),
            # dict(type='CenterShift', apply_z=False),
            # dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('color', 'normal'))
        ],
        test_mode=False,
        loop=8),
    val=dict(
        type='KULDataset',
        split='val',
        data_root=data_root,
        transform=[
            # dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.04,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            # dict(type='CenterShift', apply_z=False),
            # dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('color', 'normal'))
        ],
        test_mode=False),
    test=dict(
        type='KULDataset',
        split='test',
        data_root=data_root,
        transform=[
            dict(type='CenterShift', apply_z=True),
            #dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.06,
                hash_type='fnv',
                mode='test',
                keys=('coord', 'color', 'normal'),
                return_grid_coord=True),
            crop=None,
            post_transform=[
                #dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('color', 'normal'))
            ],
            aug_transform = [
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ])))
            # aug_transform=[[{
            #     'type': 'RandomRotateTargetAngle',
            #     'angle': [0],
            #     'axis': 'z',
            #     'center': [0, 0, 0],
            #     'p': 1
            # }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [0.5],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [1],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [1.5],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [0],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }, {
            #                    'type': 'RandomScale',
            #                    'scale': [0.95, 0.95]
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [0.5],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }, {
            #                    'type': 'RandomScale',
            #                    'scale': [0.95, 0.95]
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [1],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }, {
            #                    'type': 'RandomScale',
            #                    'scale': [0.95, 0.95]
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [1.5],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }, {
            #                    'type': 'RandomScale',
            #                    'scale': [0.95, 0.95]
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [0],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }, {
            #                    'type': 'RandomScale',
            #                    'scale': [1.05, 1.05]
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [0.5],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }, {
            #                    'type': 'RandomScale',
            #                    'scale': [1.05, 1.05]
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [1],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }, {
            #                    'type': 'RandomScale',
            #                    'scale': [1.05, 1.05]
            #                }],
            #                [{
            #                    'type': 'RandomRotateTargetAngle',
            #                    'angle': [1.5],
            #                    'axis': 'z',
            #                    'center': [0, 0, 0],
            #                    'p': 1
            #                }, {
            #                    'type': 'RandomScale',
            #                    'scale': [1.05, 1.05]
            #                }], [{
            #                    'type': 'RandomFlip',
            #                    'p': 1
            #                }]])))
