# Script to evaluate the weights for layers in a convolutional network.
#
# by M.J.Mollema

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser(description='Weight evaluations for monodepth model.')

parser.add_argument('--weight_path',    type=str, help='path to weights of a layer',            required=True)
parser.add_argument('--eval_type',      type=str, help='type of evaluation function to use',    default='mean_abs')

args = parser.parse_args()


def load_weights(weight_path: str) -> np.ndarray:
    w = np.load(weight_path)
    return w


def save_weights(w: np.ndarray, weight_path: str) -> None:
    path_split  = weight_path.split('/')
    save_folder = f'/{path_split[1]}/{path_split[2]}/{path_split[3]}/{path_split[4]}/{path_split[5]}/pruned'
    print(save_folder, path_split)
    save_path   = f'{save_folder}/{path_split[-1].split(".")[0]}.npy'
    if not Path(save_folder).exists():
        Path(save_folder).mkdir(parents=True)
    np.save(save_path, w)
    return


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


def eval_mean_l2(w: np.ndarray) -> np.ndarray:
    mean_l2_array = np.zeros((w.shape[2], w.shape[3]))
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w_mean_l2 = np.mean(w[:, :, i, j]**2)
            mean_l2_array[i, j] = w_mean_l2
    return mean_l2_array


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


def eval_sum_l2(w: np.ndarray) -> np.ndarray:
    sum_l2_array = np.zeros((w.shape[2], w.shape[3]))
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w_sum_l2 = np.sum(w[:, :, i, j]**2)
            sum_l2_array[i, j] = w_sum_l2
    return sum_l2_array


def prune(w: np.ndarray, eval_array: np.ndarray) -> np.ndarray:
    std     = np.std(eval_array)
    mean    = np.mean(eval_array)
    k       = 1

    print(np.sum(w))
    w_pruned = np.copy(w)
    to_prune_x, to_prune_y = np.where(eval_array < mean - k * std)
    for x in to_prune_x:
        for y in to_prune_y:
            w_pruned[ :, :, x, y] = 0
    print(np.sum(w_pruned))
    return w_pruned


def main():
    w = load_weights(args.weight_path)
    if args.eval_type == 'mean':
        eval_array = eval_mean(w)
    elif args.eval_type == 'mean_abs':
        eval_array = eval_mean_abs(w)
    elif args.eval_type == 'mean_l2':
        eval_array = eval_mean_l2(w)
    elif args.eval_type == 'sum':
        eval_array = eval_sum(w)
    elif args.eval_type == 'sum_abs':
        eval_array = eval_sum_abs(w)
    elif args.eval_type == 'sum_l2':
        eval_array = eval_sum_l2(w)
    else:
        eval_array = []
        print('\nPlease enter one of the following options for --eval_type:\n'
        'mean\n'
        'mean_abs\n'
        'mean_l2\n'
        'sum\n'
        'sum_abs\n'
        'sum_l2\n')
        quit()
    w_pruned = prune(w, eval_array)
    print(np.sum(w))
    save_weights(w_pruned, args.weight_path)

    # print('Evaluation result:\n'
    #       '{}'.format(eval_array))
    # print(eval_array.shape)
    # plt.plot(eval_array.flatten())
    # plt.show()


if __name__ == '__main__':
    main()
