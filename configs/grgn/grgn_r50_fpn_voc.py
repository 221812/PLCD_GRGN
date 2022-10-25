_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_cocofmt.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

use_mscrop_aug = False

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    roi_head=dict(
        type='GuideGCNRoIHead',
        bbox_head=dict(
            type='GRGNBBoxHead',
            num_classes=20,
            use_attention=True, 
            att_group=16,
            cls_gcn_cfg = dict(
                use_gcn=True,
                adj_mode='A_img_lvl_eye',
                ),
            reg_gcn_cfg = dict(
                use_gcn=True,
                adj_mode='A_img_lvl_eye',
                ),
            gcn_cfg = dict(
                use_stop_grad=True,
                num=2,
                bias=False,
                act='leakyrelu',
                norm='GN',
                fusion_mode='dot',
                emb_mode='label',
                emb_channel=768,
                adj_path='./data/VOCdevkit/adjacency_info.pkl',
                emb_path='./data/VOCdevkit/cls_embedding.pkl',
                )
            )),)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

if use_mscrop_aug:
    data = dict(
        samples_per_gpu=24,
        workers_per_gpu=1,
        train=dict(
            pipeline=train_pipeline))
else:
    data = dict(
        samples_per_gpu=24,
        workers_per_gpu=1,)

fp16 = dict(loss_scale=dict(init_scale=512))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# logger
evaluation = dict(interval=1, classwise=True, metric='bbox',save_best='bbox_mAP') # val log / epoch
log_config = dict(interval=50)  # train loss log / iter
checkpoint_config = dict(interval=20)  # model log / epoch
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8,11])
# runtime setting
runner = dict(type='EpochBasedRunner', max_epochs=12)
find_unused_parameters = True