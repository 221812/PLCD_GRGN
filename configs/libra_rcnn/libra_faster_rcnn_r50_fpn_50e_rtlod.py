_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/rtlod1300.py',
    '../_base_/schedules/schedule_100e.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
    ],
    roi_head=dict(
        bbox_head=dict(
            num_classes=23,
            loss_bbox=dict(
                _delete_=True,
                type='BalancedL1Loss',
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(sampler=dict(neg_pos_ub=5), allowed_border=-1),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='CombinedSampler',
                num=512,
                pos_fraction=0.25,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)))))

data = dict(
    samples_per_gpu=16,
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