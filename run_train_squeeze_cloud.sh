#!/bin/sh
python monodepth_main.py --mode train \
--model_name squeeze_net \
--encoder squeeze_net \
--data_path /home/shared/data/KITTI/ \
--filenames_file utils/filenames/kitti_train_files.txt \
--log_directory logs/ \
--num_epochs 50 \
--batch_size 8
# --checkpoint_path models/squeeze_net/model-squeeze_net \
