#!/bin/sh

# Script to run all training, testing, evaluation of MonoDepth with pruning

# Config
MODEL_NAME=pruning

# For summaries, etc.
LOG_DIRECTORY=logs/

# Testing
TEST_DATA_PATH=/home/shared/data/KITTI/
TEST_FILENAMES_FILE=utils/filenames/kitti_test_files_png.txt

# Evaluation
EVAL_DATA_PATH=/home/shared/data/KITTI/stereo_2015/

# For disparities
DISP_PATH=disparities/

# For pickle
RESULTS_PATH=logs/

for SPARSITY in 0.3 0.5 0.7 0.9
do

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
    --gt_path ${EVAL_DATA_PATH} \
    --description ${MODEL_NAME}_${SPARSITY}

done
