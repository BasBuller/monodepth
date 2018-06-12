#!/bin/sh

# Script to run all training, testing, evaluation of MonoDepth with pruning

# Config
MODEL_NAME=pruning

# For summaries, disparities, etc.
LOG_DIRECTORY=logs/

# Training
TRAIN_DATA_PATH=/home/shared/data/KITTI/
TRAIN_FILENAMES_FILE=utils/filenames/kitti_train_files_26_png.txt

# Testing
TEST_DATA_PATH=/home/shared/data/KITTI/
TEST_FILENAMES_FILE=utils/filenames/kitti_test_files_png.txt

# For disparities
DISP_PATH=disparities/

# For pickle
RESULTS_PATH=logs/

# Checkpoint model
CHECKPOINT_PATH=models/model_kitti

for SPARSITY in 0.3 0.5 0.7 0.9
do
    echo "Training ${SPARSITY} sparsity..."
    python monodepth_main.py --mode train \
    --model_name ${MODEL_NAME}_${SPARSITY} \
    --data_path ${TRAIN_DATA_PATH} \
    --filenames_file ${TRAIN_FILENAMES_FILE} \
    --use_prunable \
    --pruning_hparams "target_sparsity=${SPARSITY}, sparsity_function_begin_step=5000, sparsity_function_end_step=15000" \
    --output_directory ${DISP_PATH} \
    --log_directory ${LOG_DIRECTORY} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --retrain \

    echo "Testing ${SPARSITY} sparsity..."
    python monodepth_main.py --mode test \
    --model_name ${MODEL_NAME}_${SPARSITY} \
    --data_path ${TEST_DATA_PATH} \
    --filenames_file ${TEST_FILENAMES_FILE} \
    --use_prunable \
    --output_directory ${DISP_PATH} \
    --log_directory ${LOG_DIRECTORY} \
    --checkpoint_path ${LOG_DIRECTORY}/${MODEL_NAME}_${SPARSITY}/${MODEL_NAME}_${SPARSITY}

    echo "Evaluating ${SPARSITY} sparsity..."
    python utils/evaluate_kitti.py \
    --split kitti \
    --predicted_disp_path ${DISP_PATH}/${MODEL_NAME}_${SPARSITY}/disparities.npy \
    --results_path ${LOG_DIRECTORY}/results.pickle \
    --gt_path ${TEST_DATA_PATH} \
    --description ${MODEL_NAME}_${SPARSITY}

done
