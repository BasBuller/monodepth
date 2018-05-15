#!/bin/sh
python monodepth_main.py --mode train \
--model_name squeeze_net \
--data_path /home/bas/Documents/monodepth_data/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path models/squeeze_net \
--encoder squeeze_net \
--full_summary