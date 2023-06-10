#!/bin/bash

pip install -v -e .
python3 -m torch.distributed.run \
--nproc_per_node=2 \
tools/train.py \
/data/BEVDet/configs/bevdet/bevdet4d-r50-depth-cbgs-TC_afterfusion_mini_6011.py \
--work-dir /data/BEVDet/work_dirs/0530_mini_TCbeforefusion_gtdepthv1_torch113 \
--seed 0 \
--launcher pytorch > 0530file.txt 2>&1 &
#--resume-from /share/home/sjtu_fangjy/py_ws/BEVDet/work_dirs/0503-bevdet4d-r50-depth-cbgs-res34-TC_afterfusion/epoch_4.pth
