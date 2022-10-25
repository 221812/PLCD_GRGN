_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/rtlod1300.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        type='CascadeRoIHead',
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=23),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=23),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=23)]))                               
            

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=2,)

# fp16 = dict(loss_scale=dict(init_scale=512))

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