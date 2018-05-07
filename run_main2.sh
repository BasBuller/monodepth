#!/bin/sh
python monodepth_main.py --mode test \
--data_path /media/huis/EMDrive/DeepLearningData/KITTI/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files.txt \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path models/model_city2kitti_resnet