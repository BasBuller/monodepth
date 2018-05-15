"""
Script to extract the weights and biases for all layers in a convolutional network.
All are saved as .npy files.

by M.J.Mollema
"""

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import argparse
import tensorflow as tf
from pathlib import Path

from monodepth_model import *

parser = argparse.ArgumentParser(description='Weight extraction for monodepth model.')

parser.add_argument('--encoder',		  type=str,   help='type of encoder, vgg or resnet50', default='vgg')
# parser.add_argument('--image_path',	   type=str,   help='path to the image', required=True)
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',	 type=int,    help='input height', default=256)
parser.add_argument('--input_width',	  type=int,   help='input width', default=512)

args = parser.parse_args()

def weight_extractor(params):
	global restore_path
	left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
	model = MonodepthModel(params, "test", left, None)

	# SESSION
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)

	# SAVER
	train_saver = tf.train.Saver()

	# INIT
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	coordinator = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

	# RESTORE
	restore_path = args.checkpoint_path.split(".")[0]
	train_saver.restore(sess, restore_path)

	# SAVE WEIGHTS TO .npy FILES
	var, names = [], []
	for v in tf.trainable_variables():
		# if v.name == "model/encoder/Conv_6/weights:0":
		# 	var.append(v)
		# 	names.append(v.name)
		var.append(v)
		names.append(v.name)
	vars = sess.run(var)
	for i in range(len(names)):
		name = names[i].split('/')
		save_folder = '{0}/{1}/{2}/'.format(restore_path.split('/')[2],
												restore_path.split('/')[3],
												name[1])
		save_path	= '{0}/{1}_{2}.npy'.format(save_folder,
												name[2],
												name[3][:-2])
		if not Path(save_folder).exists():
			Path(save_folder).mkdir(parents=True)
		np.save(save_path, vars[i])


	print('done!')

def main(_):

	params = monodepth_parameters(
		encoder=args.encoder,
		height=args.input_height,
		width=args.input_width,
		batch_size=2,
		num_threads=1,
		num_epochs=1,
		do_stereo=False,
		wrap_mode="border",
		use_deconv=False,
		alpha_image_loss=0,
		disp_gradient_loss_weight=0,
		lr_loss_weight=0,
		full_summary=False)

	weight_extractor(params)

if __name__ == '__main__':
	tf.app.run()