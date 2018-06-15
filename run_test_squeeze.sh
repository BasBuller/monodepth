#!/bin/bash
python monodepth_main.py --mode test \
--data_path data/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--log_directory logs/ \
--checkpoint_path models/squeeze_net/model-squeeze \
--output_directory results/