#!/bin/sh
python monodepth_main.py --mode train \
--model_name pruning_05 \
--data_path /home/shared/data/KITTI/ \
--filenames_file utils/filenames/kitti_train_files.txt \
--use_prunable \
--pruning_hparams target_sparsity=0.5 \
--output_directory disparities/ \
--log_directory logs/ \
