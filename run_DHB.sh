#!/bin/bash
python ./script/run/dist_infer_NL.py --output ./log/4DGS_Exper_macpool_v3/wo_mlp/NL_test1.txt \
        --dataset DHB \
        --dataset_path ./data/DHB-dataset \
        --scenes_list ./data/DHB_scene_list_test.txt \
        --num_points 1024 \
        --iters 1000 \
        --num_gpus 8 \
        --max_processes_per_gpu 2 \
> log/DHB_longdress_test_iter100- 2>&1 &
# --demo
