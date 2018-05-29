#!/bin/sh
python monodepth_main.py --mode train \
--model_name insert_model_name \
--data_path /home/shared/KITTI/ \
--filenames_file utils/filenames/2011_09_29_files.txt \
--use_prunable \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path /home/shared/models/model_city2kitti \
--retrain \
--full_summary