#!/bin/sh
python monodepth_main.py --mode train \
--model_name 1day_cloud \
--data_path ~/KITTI/ \
--filenames_file utils/filenames/2011_09_29_files.txt \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path models/1day_cloud \
--retrain \
--full_summary