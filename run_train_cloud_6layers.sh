#!/bin/sh
python monodepth_main.py --mode train \
--model_name 6layers_kitti_1day \
--data_path /home/shared/KITTI/ \
--filenames_file utils/filenames/2011_09_29_files.txt \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path /home/shared/models/model_city2kitti \
--retrain \
--full_summary \
--num_layers 6
