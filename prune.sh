#!/bin/bash

EVAL_TYPE=mean_abs
OUTPUT_DIR=/monodepth/models/city2kitti/pruned
NUM_STD=10

python weight_eval.py --weight_path=/monodepth/models/city2kitti/encoder/Conv/weights:0.npy --eval_type=$EVAL_TYPE --output_dir=$OUTPUT_DIR --num_std=$NUM_STD;
for i in {1..13};
do
    python weight_eval.py --weight_path=/monodepth/models/city2kitti/encoder/Conv_$i/weights:0.npy --eval_type=$EVAL_TYPE --output_dir=$OUTPUT_DIR --num_std=$NUM_STD;
done

python weight_eval.py --weight_path=/monodepth/models/city2kitti/decoder/Conv/weights:0.npy --eval_type=$EVAL_TYPE --output_dir=$OUTPUT_DIR --num_std=$NUM_STD;
for i in {1..17};
do
    python weight_eval.py --weight_path=/monodepth/models/city2kitti/decoder/Conv_$i/weights:0.npy --eval_type=$EVAL_TYPE --output_dir=$OUTPUT_DIR --num_std=$NUM_STD;
done
