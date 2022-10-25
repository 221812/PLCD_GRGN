_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/rtlod300.py',
    '../_base_/schedules/schedule_140e.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
    bbox_head=[
        dict(type='Shared2FCBBoxHead',num_classes=13),
        dict(type='Shared2FCBBoxHead',num_classes=13),
        dict(type='Shared2FCBBoxHead',num_classes=13),]))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# logger
evaluation = dict(interval=10, metric='bbox') # val log / epoch
log_config = dict(interval=10)         # train loss log / iter
checkpoint_config = dict(interval=30)  # model log / epoch