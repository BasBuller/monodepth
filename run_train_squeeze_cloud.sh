#!/bin/sh
python monodepth_main.py --mode train \
--model_name squeeze_net \
--encoder squeeze_net \
--data_path /home/shared/KITTI/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--output_directory disparities/squeeze_net/ \
# --checkpoint_path dispartities/squeeze_net/ \
--log_directory logs/ \
--num_epochs 1 \
--batch_size 20 \
--full_summary
