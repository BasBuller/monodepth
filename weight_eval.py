# Script to evaluate the weights for layers in a convolutional network.
#
# by M.J.Mollema

import argparse
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser(description='Weight evaluations for monodepth model.')

parser.add_argument('--weight_path',    type=str, help='path to weights of a layer',            required=True)
parser.add_argument('--eval_type',      type=str, help='type of evaluation function to use',    default='mean_abs')

args = parser.parse_args()


def load_weights(weight_path: str):
    w: np.ndarray = np.load(weight_path)
    return w


def eval_mean(w: np.ndarray) -> np.ndarray:
    mean_array = np.zeros((w.shape[2], w.shape[3]))
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w_mean = np.mean(w[:, :, i, j])
            mean_array[i, j] = w_mean
    return mean_array


def eval_mean_abs(w: np.ndarray) -> np.ndarray:
    mean_abs_array = np.zeros((w.shape[2], w.shape[3]))
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w_mean_abs = np.mean(np.abs(w[:, :, i, j]))
            mean_abs_array[i, j] = w_mean_abs
    return mean_abs_array


def eval_sum(w: np.ndarray) -> np.ndarray:
    sum_array = np.zeros((w.shape[2], w.shape[3]))
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w_sum = np.sum(w[:, :, i, j])
            sum_array[i, j] = w_sum
    return sum_array


def eval_sum_abs(w: np.ndarray) -> np.ndarray:
    sum_abs_array = np.zeros((w.shape[2], w.shape[3]))
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w_sum_abs = np.sum(np.abs(w[:, :, i, j]))
            sum_abs_array[i, j] = w_sum_abs
    return sum_abs_array


def eval_cos_sim(w: np.ndarray) -> np.ndarray:
    cos_sim_array = np.zeros((w.shape[2], w.shape[3]))
    SIMc = 


def main():
    w = load_weights(args.weight_path)
    if args.eval_type == 'mean':
        eval_array = eval_mean(w)
    elif args.eval_type == 'mean_abs':
        eval_array = eval_mean_abs(w)
    elif args.eval_type == 'sum':
        eval_array = eval_sum(w)
    elif args.eval_type == 'sum_abs':
        eval_array = eval_sum_abs(w)
    else:
        eval_array = []
        print('\nPlease enter one of the following options for --eval_type:\n'
              'mean\n'
              'mean_abs\n'
              'sum\n'
              'sum_abs\n')
        quit()
    # print('Evaluation result:\n'
    #       '{}'.format(eval_array))
    print(eval_array.shape)
    plt.plot(eval_array.flatten())
    plt.show()

if __name__ == '__main__':
    main()
