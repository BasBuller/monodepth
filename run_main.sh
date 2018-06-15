#!/bin/sh
for MODEL in squeeze_net delayed_pool small_decoder
do
    echo "Running ${MODEL}..."
    python monodepth_main.py --mode test \
    --data_path ~/Documents/monodepth/data/stereo_2015/ \
    --filenames_file utils/filenames/kitti_stereo_2015_test_files_png.txt \
    --output_directory disparities/$MODEL/ \
    --log_directory logs/$MODEL/ \
    --checkpoint_path models/$MODEL/$MODEL \
    --full_summary
done