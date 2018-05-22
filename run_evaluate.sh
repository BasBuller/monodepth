#!/bin/sh
python utils/evaluate_kitti.py \
--split kitti \
--predicted_disp_path disparities/insert_model_name_095/disparities.npy \
--results_path results/insert_model_name_095/results.pickle \
--gt_path /media/huis/EMDrive/DeepLearningData/KITTI/stereo_2015/