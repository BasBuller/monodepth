#!/bin/sh
python monodepth_main.py --mode train \
--model_name insert_model_name \
--data_path /media/huis/EMDrive/DeepLearningData/KITTI/ \
--filenames_file utils/filenames/kitti_train_files_random_png.txt \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path models/model_city2kitti \
--retrain \
--full_summary