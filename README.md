# PLCD_GRGN
This repo is the implementation of [GRGN](https://ieeexplore.ieee.org/document/9976247).

# Installation
Please refer to the [Installation](https://mmdetection.readthedocs.io/en/stable/get_started.html) of mmdetection. The version of mmdetection is 2.20.0.

- Prepare MMDET env
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

- Add custom modules
    - [Custom roi_head](https://github.com/221812/PLCD_GRGN/blob/master/mmdet/models/roi_heads/guidegcn_roi_head.py) and corresponding [init.py](https://github.com/221812/PLCD_GRGN/blob/master/mmdet/models/roi_heads/__init__.py)
    - [Custom bbox_head](https://github.com/221812/PLCD_GRGN/blob/master/mmdet/models/roi_heads/bbox_heads/grgn_bbox_head.py) and corresponding [init.py](https://github.com/221812/PLCD_GRGN/blob/master/mmdet/models/roi_heads/bbox_heads/__init__.py)
    - [Custom config](https://github.com/221812/PLCD_GRGN/blob/master/configs/grgn/grgn_r50_fpn_voc.py)

# Usages
- Generate co-occur information
```
python tools/dataset_converters/co-occur.py
```
- Train baseline
```
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh \
    configs/grgn/baseline_r50_fpn_voc.py \
    2 \
    --work-dir logs/voc0712/baseline \
```
- Train GRGN
```
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh \
    configs/grgn/grgn_r50_fpn_voc.py \
    2 \
    --work-dir logs/voc0712/grgn \
```
- Test 
```
bash tools/dist_test.sh \
    logs/voc0712/grgn/grgn_r50_fpn_voc.py \
    logs/voc0712/grgn/latest.pth \
    1 \
    --eval bbox \
```

# Acknowledgement
Our implementation is mainly based on the codebase of [mmdetection](https://github.com/open-mmlab/mmdetection). We gratefully thank the authors for their wonderful works.
