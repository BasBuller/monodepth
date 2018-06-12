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

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()

def save_weights(var, names, sess, prune=True):
    vars = sess.run(var)
    for i in range(len(names)):
        name = names[i].split('/')
        save_folder = '{0}/{1}/{2}/{3}'.format(restore_path.split('/')[2],
                                               restore_path.split('/')[3],
                                               name[1],
                                               name[2])
        if prune:
            save_path = '{0}/{1}.npy'.format(save_folder,
                                            name[3])
        else:
            save_path = f'{save_folder}/{name[3]}_keep.npy'
        if not Path(save_folder).exists():
            Path(save_folder).mkdir(parents=True)
        np.save(save_path, vars[i])


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
    # threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    # SAVE WEIGHTS TO .npy FILES
    var_prune, names_prune, var_keep, names_keep = [], [], [], []
    prune_weights = ["model/encoder/Conv_1/weights:0",
                    "model/encoder/Conv_3/weights:0",
                    "model/encoder/Conv_5/weights:0",
                    "model/encoder/Conv_7/weights:0",
                    "model/encoder/Conv_9/weights:0",
                    "model/encoder/Conv_11/weights:0",
                    "model/encoder/Conv_13/weights:0",
                    "model/decoder/Conv/weights:0",
                    "model/decoder/Conv_2/weights:0",
                    "model/decoder/Conv_4/weights:0",
                    "model/decoder/Conv_6/weights:0",
                    "model/decoder/Conv_9/weights:0",
                    "model/decoder/Conv_12/weights:0",
                    "model/decoder/Conv_15/weights:0",
                    "model/encoder/Conv_1/biases:0",
                    "model/encoder/Conv_3/biases:0",
                    "model/encoder/Conv_5/biases:0",
                    "model/encoder/Conv_7/biases:0",
                    "model/encoder/Conv_9/biases:0",
                    "model/encoder/Conv_11/biases:0",
                    "model/encoder/Conv_13/biases:0",
                    "model/decoder/Conv/biases:0",
                    "model/decoder/Conv_2/biases:0",
                    "model/decoder/Conv_4/biases:0",
                    "model/decoder/Conv_6/biases:0",
                    "model/decoder/Conv_9/biases:0",
                    "model/decoder/Conv_12/biases:0",
                    "model/decoder/Conv_15/biases:0"]

    for v in tf.trainable_variables():
        if v.name in prune_weights:
            var_prune.append(v)
            names_prune.append(v.name)
        elif v.name not in prune_weights:
            var_keep.append(v)
            names_keep.append(v.name)
        # var.append(v)
        # names.append(v.name)
    save_weights(var_prune, names_prune, sess, prune=True)
    save_weights(var_keep, names_keep, sess, prune=False)

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
