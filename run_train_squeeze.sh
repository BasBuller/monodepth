#!/bin/sh


python monodepth_main.py --mode train \
--model_name squeeze_net \
--encoder squeeze_net \
--num_threads 1 \
--retrain \
--num_epochs 1 \
--batch_size 20 \
--data_path data/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--output_directory disparities/ \
--log_directory logs/ \
# --checkpoint_path models/squeeze_net \
#--input_height 375 \
#--input_width 1242 
#--full_summary 

