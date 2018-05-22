#!/bin/sh
python utils/evaluate_kitti.py \
--split kitti \
--predicted_disp_path disparities/disparities.npy \
--results_path /home/shared/results/results.pickle \
--gt_path /home/shared/KITTI/stereo_2015/ \
--description 5layers_retrained
