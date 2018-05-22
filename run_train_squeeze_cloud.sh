#!/bin/sh
python monodepth_main.py --mode train \
--model_name squeeze_net \
--encoder squeeze_net \
--data_path /home/shared/KITTI/stereo_2015/ \
--filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
--log_directory logs/ \
--checkpoint_path models/squeeze_net/model-squeeze_net \
--retrain \
--num_epochs 1 \
--batch_size 20 \

# --full_summary
# --output_directory disparities/ \
# --checkpoint_path /home/shared/models/squeeze_net \
# --retrain \