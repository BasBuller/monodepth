#!/bin/sh
python monodepth_main.py --mode train \
--model_name small_decoder_two \
--encoder small_decoder_two \
--data_path data/stereo2015 \
--filenames_file utils/filenames/kitti_train_files_random_png.txt \
--output_directory disparities/ \
--log_directory logs/ \
--checkpoint_path models/small_decoder_two/small_decoder_two \
--retrain 

