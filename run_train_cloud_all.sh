#!/bin/sh

# Script to run all training and testing of MonoDepth with layers removed

# Config
MODEL_NAME=monodepth_kitti
LOG_DIRECTORY=logs/

#TRAIN_FILENAMES_FILE=utils/filenames/2011_09_29_files.txt
TRAIN_FILENAMES_FILE=utils/filenames/kitti_train_files_png.txt
#CHECKPOINT_PATH=/home/shared/models/model_kitti
#TRAIN_DATA_PATH=/home/shared/KITTI/
TRAIN_DATA_PATH=/home/shared/data/KITTI/

#TEST_DATA_PATH=/home/shared/KITTI/stereo_2015/
TEST_DATA_PATH=/home/shared/data/KITTI/
#TEST_FILENAMES_FILE=utils/filenames/kitti_stereo_2015_test_files_png.txt
TEST_FILENAMES_FILE=utils/filenames/kitti_test_files_png.txt

RESULTS_PATH=/home/shared/results/results.pickle

for MODEL in delayed_pool_two small_decoder_two
do
    echo "Training ${MODEL}..."
    python monodepth_main.py --mode train \
    --model_name $MODEL \
    --encoder $MODEL \
    --data_path  $TRAIN_DATA_PATH \
    --filenames_file $TRAIN_FILENAMES_FILE \
    --log_directory $LOG_DIRECTORY \
    --num_epochs 50 \
    --batch_size 8

    echo "Training ${MODEL}..."
    python monodepth_main.py --mode train \
    --model_name $MODEL \
    --encoder $MODEL \
    --data_path  $TRAIN_DATA_PATH \
    --filenames_file $TRAIN_FILENAMES_FILE \
    --log_directory $LOG_DIRECTORY \
    --num_epochs 50 \
    --batch_size 8
done
