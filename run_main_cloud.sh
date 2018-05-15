#!/bin/sh
python monodepth_main.py --mode test \
--data_path /home/shared/KITTI/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path models/1day_cloud_950 \
--full_summary