#!/bin/sh
python monodepth_main.py --mode test \
--model_name insert_model_name \
--data_path /home/shared/KITTI/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path models/insert_model_name \
--full_summary