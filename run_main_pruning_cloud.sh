#!/bin/sh
python monodepth_main.py --mode test \
--model_name insert_model_name_095 \
--data_path /home/shared/KITTI/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--use_prunable \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path logs/insert_model_name_05/insert_model_name_05-950 \
--full_summary