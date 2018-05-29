#!/bin/sh
python monodepth_main.py --mode train \
--model_name small_decoder_one \
--encoder small_decoder_one \
--data_path /home/shared/KITTI/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--log_directory logs/ \
--num_epochs 1 \
--batch_size 10
# --checkpoint_path models/squeeze_net/model-squeeze_net \