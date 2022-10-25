# PLCD_GRGN
This repo is the implementation of the under review paper.
The detailed usages is coming soon...

# Installation
Please refer to the [Installation](https://mmdetection.readthedocs.io/en/stable/get_started.html) of mmdetection. The version of mmdetection is 2.20.0.

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
