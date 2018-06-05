# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.model_pruning as pruning
from tensorflow.python import pywrap_tensorflow

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', required=True)
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--use_prunable',                          help='if set, will use prunable convolutions', action='store_true')
parser.add_argument('--pruning_hparams',           type=str,   help='comma separated list of pruning-related hyperparameters', default='')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--quantize',                              help='if set, will write a ProtoBuf containing the graph to the checkpoint_path', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def print_tensors_in_checkpoint_file(file_name, all_tensors=True):

    var_list = []
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)

    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()

      for key in sorted(var_to_shape_map):
        var_list.append(key)

    return var_list

def train(params):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print(f"total number of samples: {num_training_samples}")
        print(f"total number of steps: {num_total_steps}")

        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch

        # split for each gpu
        left_splits  = tf.split(left,  args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)

        tower_grads  = []
        tower_losses = []
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):

                    model = MonodepthModel(params, args.mode, left_splits[i], right_splits[i], reuse_variables, i)

                    loss = model.total_loss
                    tower_losses.append(loss)

                    reuse_variables = True

                    grads = opt_step.compute_gradients(loss)

                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, collections=['model_0'])
        tf.summary.scalar('total_loss', total_loss, collections=['model_0'])
        # summary_op = tf.summary.merge([
        #     tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES),
        #     tf.summary.merge_all(key='model_0')], collections='merged')

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)

        if args.use_prunable:
            var_to_restore_names = print_tensors_in_checkpoint_file(args.checkpoint_path.split(".")[0])
            var_to_restore = [v for v in tf.global_variables() if v.name.split(":")[0] in var_to_restore_names]
            train_loader = tf.train.Saver(var_to_restore)
            train_saver = tf.train.Saver()
        else:
            train_loader = tf.train.Saver()
            train_saver = tf.train.Saver()

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()

        print(f"number of trainable parameters: {total_num_parameters}")
        print(f"trainable parameters: {tf.trainable_variables()}")

        # PRUNING
        if args.use_prunable:

            # Parse pruning hyperparameters
            pruning_hparams = pruning.get_pruning_hparams().parse(args.pruning_hparams)

            # Create a pruning object using the pruning hyperparameters
            pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)

            # Use the pruning object to add ops to the training graph to update the masks
            # The conditional_mask_update_op will update the masks only when the
            # training step is in [begin_pruning_step, end_pruning_step] specified in
            # the pruning spec proto
            # mask_update_op = pruning_obj.conditional_mask_update_op()
            mask_update_op = pruning_obj.mask_update_op()

            # Use the pruning object to add summaries to the graph to track the sparsity
            # of each of the layers
            pruning_obj.add_pruning_summaries()

        # DEFINE SUMMARY OP
        summary_op = tf.summary.merge([
            tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES),
            tf.summary.merge_all(key='model_0')], collections='merged')

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_loader.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()

            if args.use_prunable and step and step % 10 == 0:

                # Print weight sparsity
                print(f"weight sparsity: {sess.run(pruning.get_weight_sparsity())}")

            _, loss_value = sess.run([apply_gradient_op, total_loss])

            if args.use_prunable:
                sess.run(mask_update_op)

            duration = time.time() - before_op_time

            if step and step % 100 == 0:

                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            if step and step % 10000 == 0:

                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/' + args.model_name, global_step=step)

        # Save in log directory
        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/' + args.model_name, global_step=num_total_steps)

def test(params):
    """Test function."""

    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = MonodepthModel(params, args.mode, left, right)

    # SUMMARY
    summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # WRITE SUMMARY
    summary_writer.add_graph(sess.graph)

    # SAVER
    train_loader = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_loader.restore(sess, restore_path)

    if args.quantize:

        # QUANTIZE GRAPH
        tf.contrib.quantize.create_eval_graph()
        print('created simulated quantized graph')

        # SAVE QUANTIZED GRAPH
        with open(args.checkpoint_path + '_quantized.pb', 'w') as f:
            f.write(str(sess.graph.as_graph_def()))
        print('saved simulated quantized graph')

        # To convert simulated quantized graph to real one, use command:
        # (no idea how it works)
        #
        # bazel build tensorflow/python/tools:freeze_graph && \
        # bazel-bin/tensorflow/python/tools/freeze_graph \
        # --input_graph=model_city2kitty_quantized.pb \
        # --input_checkpoint=model_city2kitty \
        # --output_graph=frozen_city2kitty.pb \
        # --output_node_names=outputs

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities    = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    for step in range(num_test_samples):
        disp = sess.run(model.disp_left_est[0])
        disparities[step] = disp[0].squeeze()
        disparities_pp[step] = post_process_disparity(disp.squeeze())

    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory + args.model_name
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)

    print('done.')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        use_prunable=args.use_prunable,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()
