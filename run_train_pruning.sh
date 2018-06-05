#!/bin/sh
python monodepth_main.py --mode train \
--model_name test_print \
--data_path ~/Downloads/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_train_files_png.txt \
--num_epochs 1 \
--use_prunable \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path models/model_city2kitti \
--retrain \
--full_summary