#!/bin/sh

# Script to run all training and testing of MonoDepth with layers removed

# Config
MODEL_NAME=monodepth_kitti
LOG_DIRECTORY=logs/

#TRAIN_FILENAMES_FILE=utils/filenames/2011_09_29_files.txt
TRAIN_FILENAMES_FILE=utils/filenames/kitti_train_files.txt
#CHECKPOINT_PATH=/home/shared/models/model_kitti
#TRAIN_DATA_PATH=/home/shared/KITTI/
TRAIN_DATA_PATH=/home/shared/data/KITTI/

#TEST_DATA_PATH=/home/shared/KITTI/stereo_2015/
TEST_DATA_PATH=/home/shared/data/KITTI/
#TEST_FILENAMES_FILE=utils/filenames/kitti_stereo_2015_test_files_png.txt
TEST_FILENAMES_FILE=utils/filenames/kitti_test_files.txt

RESULTS_PATH=/home/shared/results/results.pickle

for LAYERS in 6 5 4
do
    echo "Training ${LAYERS} layers..."
    python monodepth_main.py --mode train \
    --model_name ${MODEL_NAME}_${LAYERS}layers \
    --data_path  $TRAIN_DATA_PATH \
    --filenames_file $TRAIN_FILENAMES_FILE \
    --output_directory disparities/ \
    --full_summary \
    --num_layers ${LAYERS}

    echo "Running ${LAYERS} layers..."
    python monodepth_main.py --mode test \
    --data_path $TEST_DATA_PATH \
    --filenames_file $TEST_FILENAMES_FILE \
    --output_directory disparities/ \
    --log_directory $LOG_DIRECTORY \
    --checkpoint_path ${LOG_DIRECTORY}/${MODEL_NAME}_${LAYERS}layers/model-950 \
    --full_summary \
    --num_layers ${LAYERS}

    echo "Evaluating ${LAYERS} layers..."
    python utils/evaluate_kitti.py \
    --split kitti \
    --predicted_disp_path disparities/disparities.npy \
    --results_path /home/shared/results/results.pickle \
    --gt_path $TEST_DATA_PATH \
    --description ${MODEL_NAME}_${LAYERS}layers
done
