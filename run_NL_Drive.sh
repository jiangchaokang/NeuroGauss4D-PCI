#!/bin/bash
python ./script/run/dist_infer_NL.py --output ./log/4DGS_Exper_macpool_v3/wo_mlp/NL_test1.txt \
        --scenes_list_file ./data/list/NL_Drive_test_1.txt \
        --layer_width 32 \
        --iters 1000 \
        --num_gpus 8 \
        --max_processes_per_gpu 2 \
> log/NL_Drive_release_test 2>&1
