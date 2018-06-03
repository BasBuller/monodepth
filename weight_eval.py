# Script to evaluate the weights for layers in a convolutional network.
#
# by M.J.Mollema
# TODO: REMOVE CORRESPONDING INPUT LAYER WHEN DELETING OUTPUT OF PREVIOUS LAYER

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser(description='Weight evaluations for monodepth model.')

parser.add_argument('--weight_path',    type=str,   help='path to weights of a layer', required=True)
parser.add_argument('--eval_type',      type=str,   help='type of evaluation function to use', default='mean_abs')
parser.add_argument('--num_std',        type=float, help='number of standard deviations to use', default=2)
parser.add_argument('--output_dir',     type=str,   help='path to output directory', required=True)

args = parser.parse_args()


def load_weights(weight_path: str) -> np.ndarray:
    w = np.load(weight_path)
    return w


def load_biases(weight_path: str) -> np.ndarray:
    bias_path = Path(f'{weight_path.parent}/biases:0.npy')
    b = np.load(bias_path)
    return b


def save_weights_biases(wb: np.ndarray, weight_path: str, output_dir: str) -> None:
    if len(wb.shape) == 4:
        filename = 'weights:0.npy'
    elif len(wb.shape) == 1:
        filename = 'bias:0.npy'
    else:
        raise ValueError(f'Got {len(wb.shape)} dimensions, expected either 4 for weights '
        f'array or 1 for bias array')
    parts = weight_path.parts
    save_folder = f'{output_dir}/{args.eval_type}_std_{args.num_std}/' \
                  f'{parts[-3]}/{parts[-2]}'
    save_path = Path(f'{save_folder}/{filename}')
    if not Path(save_folder).exists():
        Path(save_folder).mkdir(parents=True)
    np.save(save_path, wb)
    print(f'Pruned {filename} saved to: {save_path}')
    return None


def save_sizes(w: np.ndarray, weight_path: str, output_dir: str) -> None:
    save_path = f'{output_dir}/{args.eval_type}_std_{args.num_std}/sizes.csv'
    try:
        df = pd.read_csv(save_path)
    except:
        df = pd.DataFrame(columns=['en/de', 'Layer', 'Shape'])
    parts = weight_path.parts
    en_de = parts[-3]
    layer = parts[-2]

    idx = df.index[(df['en/de'] == en_de) & (df['Layer'] == layer)]
    data = {'en/de': en_de, 'Layer': layer, 'Shape': w.shape[-1]}
    if idx.empty:
        df = df.append(data, ignore_index=True)
    else:
        df.iloc[idx[0]] = data
    df.to_csv(save_path, index=False)
    return None


def eval_mean(w: np.ndarray) -> np.ndarray:
    mean_array = np.zeros(w.shape[-1])
    for i in range(len(mean_array)):
        w_mean = np.mean(w[:, :, :, i])
        mean_array[i] = w_mean
    return mean_array


def eval_mean_abs(w: np.ndarray) -> np.ndarray:
    mean_abs_array = np.zeros(w.shape[-1])
    for i in range(len(mean_abs_array)):
        w_mean_abs = np.mean(np.abs(w[:, :, :, i]))
        mean_abs_array[i] = w_mean_abs
    return mean_abs_array


def eval_mean_l2(w: np.ndarray) -> np.ndarray:
    mean_l2_array = np.zeros(w.shape[-1])
    for i in range(len(mean_l2_array)):
        w_mean_l2 = np.mean(w[:, :, :, i]**2)
        mean_l2_array[i] = w_mean_l2
    return mean_l2_array


def eval_sum(w: np.ndarray) -> np.ndarray:
    sum_array = np.zeros(w.shape[-1])
    for i in range(len(sum_array)):
        w_sum = np.sum(w[:, :, :, i])
        sum_array[i] = w_sum
    return sum_array


def eval_sum_abs(w: np.ndarray) -> np.ndarray:
    sum_abs_array = np.zeros(w.shape[-1])
    for i in range(len(sum_abs_array)):
        w_sum_abs = np.sum(np.abs(w[:, :, :, i]))
        sum_abs_array[i] = w_sum_abs
    return sum_abs_array


def eval_sum_l2(w: np.ndarray) -> np.ndarray:
    sum_l2_array = np.zeros(w.shape[-1])
    for i in range(len(sum_l2_array)):
        w_sum_l2 = np.sum(w[:, :, :, i]**2)
        sum_l2_array[i] = w_sum_l2
    return sum_l2_array


def prune(w: np.ndarray, b: np.ndarray, eval_array: np.ndarray) -> np.ndarray:
    std     = np.std(eval_array)
    mean    = np.mean(eval_array)
    k       = args.num_std

    if len(eval_array) == 2:
        to_prune = []
    else:
        to_prune = np.where(eval_array < mean - k * std)
    w_pruned = np.delete(w, to_prune, -1)
    b_pruned = np.delete(b, to_prune, -1)

    print(f'\nMean = {mean:.5f}\nSTD  = {std:.5f}')
    print(f'Removed filters: {len(to_prune[0])} ({len(to_prune[0])/len(eval_array)*100}%)\n')
    return w_pruned, b_pruned


def main():
    weight_path = Path(args.weight_path)
    output_dir = Path(args.output_dir)

    w = load_weights(weight_path)
    b = load_biases(weight_path)

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

    w_pruned, b_pruned = prune(w, b, eval_array)

    save_weights_biases(w_pruned, weight_path, output_dir)
    save_weights_biases(b_pruned, weight_path, output_dir)
    save_sizes(w_pruned, weight_path, output_dir)


if __name__ == '__main__':
    main()
