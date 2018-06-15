#!/bin/sh
python monodepth_main.py --mode train \
--model_name small_decoder_two \
--encoder small_decoder_two \
--data_path /home/shared/data/KITTI/ \
--filenames_file utils/filenames/kitti_train_files.txt \
--log_directory logs/ \
--num_epochs 50 \
--batch_size 8
# --checkpoint_path models/squeeze_net 