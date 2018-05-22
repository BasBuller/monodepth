#!/bin/sh
python monodepth_main.py --mode test \
--model_name insert_model_name_095 \
--data_path /media/huis/EMDrive/DeepLearningData/KITTI/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--use_prunable \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path ~/Projects/DeepLearningProject/cloud_files/insert_model_name_095/insert_model_name_095-950 \
--full_summary