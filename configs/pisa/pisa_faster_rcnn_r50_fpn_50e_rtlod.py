_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/rtlod1300.py',
    '../_base_/schedules/schedule_100e.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        type='PISARoIHead',
        bbox_head=dict(
            num_classes=23,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            sampler=dict(
                type='ScoreHLRSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                k=0.5,
                bias=0.),
            isr=dict(k=2, bias=0),
            carl=dict(k=1, bias=0.2))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)))

data = dict(
    samples_per_gpu=24,
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