#!/bin/sh
python ~/Projects/DeepLearningProject/monodepth/monodepth_main.py --mode test \
--data_path ~/Projects/DeepLearningProject/monodepth/data/KITTI_stereo_2015_left/ \
--filenames_file ~/Projects/DeepLearningProject/monodepth/data/KITTI_stereo_2015_left_filenames.txt \
--output_directory ~/Projects/DeepLearningProject/monodepth/output
--log_directory ~/Projects/DeepLearningProject/monodepth/logs \
--checkpoint_path ~/Projects/DeepLearningProject/monodepth/models/model_city2kitti