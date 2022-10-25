_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/rtlod1300.py',
    '../_base_/schedules/schedule_100e.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        type='DoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            _delete_=True,
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=23,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))))

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,)

# optimizer
optimizer = dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=0.0001)
# logger
evaluation = dict(interval=1, classwise=True, metric='bbox',save_best='bbox_mAP') # val log / epoch
log_config = dict(interval=10)           # train loss log / iter
checkpoint_config = dict(interval=1000)  # model log / epoch
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7,9])
# runtime setting
runner = dict(type='EpochBasedRunner', max_epochs=10) # 10*5=50
