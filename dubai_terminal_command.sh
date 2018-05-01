#!/bin/bash
python3 monodepth_main.py --mode test --data_path ~/Documents/images_monodepth/dubai_test/ --filenames_file ~/Documents/images_monodepth/dubai_test_files.txt --output_directory ~/Documents/images_monodepth/dubai_results/ --checkpoint_path ~/Documents/monodepth/models/model_cityscapes

# --log_directory ~/Documents/images_monodepth/dubai_results/