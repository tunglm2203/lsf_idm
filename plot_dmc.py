import argparse
import ast
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob2
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, nargs='+')
parser.add_argument('--radius', type=int, default=0)
parser.add_argument('--range', type=int, default=-1, help='Number of transitions want to plot')
parser.add_argument('--legend', type=str, default='', nargs='+')
parser.add_argument('--title', type=str, default='')
parser.add_argument('--shaded_std', type=bool, default=True)
parser.add_argument('--shaded_err', type=bool, default=False)
parser.add_argument('--train_test', action='store_true')

parser.add_argument('--env', type=str, default='cheetah_run')
parser.add_argument('--drq_dir', type=str, default='/home/tung/workspace/rlbench/drq/')
parser.add_argument('--plot_drq', action='store_true')
parser.add_argument('--slac_dir', type=str, default='/home/tung/workspace/rlbench/slac/')
parser.add_argument('--plot_slac', action='store_true')

args = parser.parse_args()


def smooth(y, radius, mode='two_sided', valid_only=False):
    # Copy from: https://github.com/openai/baselines/blob/master/baselines/common/plot_util.py
    '''
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


def get_info_env(path):
    info = dict(
        domain_name=None,
        task_name=None,
        action_repeat=None,
    )
    json_file = os.path.join(path, 'args.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    for k in info.keys():
        info[k] = data[k]
    return info


def get_data_in_subdir(parent_path, x_key, y_key):
    child_paths = [os.path.abspath(os.path.join(path, '..'))
                   for path in glob2.glob(os.path.join(parent_path, '**', 'eval.log'))]

    data_x, data_y = [], []
    for path in child_paths:
        json_file = os.path.join(path, 'eval.log')
        data = []
        for line in open(json_file, 'r'):
            data.append(json.loads(line))

        len_data = len(data)
        x, y = [], []
        for i in range(len_data):
            x.append(data[i][x_key])
            y.append(data[i][y_key])
        x = np.array(x)
        y = np.array(y)
        y = smooth(y, radius=args.radius)
        data_x.append(x)
        data_y.append(y)

    info_env = get_info_env(child_paths[0])
    task_name = info_env['domain_name'] + '-' + info_env['task_name']

    return (data_x, data_y), task_name, info_env


def get_values_with_range(xs, ys, truncate):
    n_experiments = len(xs)
    _xs = []
    _ys = []
    for k in range(n_experiments):
        found_idxes = np.argwhere(xs[k] >= truncate)
        if len(found_idxes) == 0:
            print("[WARNING] Last index is {}, consider choose smaller range in {}".format(
                xs[k][-1], directories[i]))
            _xs.append(xs[k][:])
            _ys.append(ys[k][:])
        else:
            range_idx = found_idxes[0, 0]
            _xs.append(xs[k][:range_idx])
            _ys.append(ys[k][:range_idx])
    return _xs, _ys


def plot_multiple_results(directories):
    x_key = 'step'
    y_key = 'mean_episode_reward'

    collect_data, plot_titles, info_envs = [], [], []
    for directory in directories:
        data_in_subdir, task_name, info_env = get_data_in_subdir(directory, x_key, y_key)
        collect_data.append(data_in_subdir)
        plot_titles.append(task_name)
        info_envs.append(info_env)

    if args.plot_drq:
        plot_drq_results()
    if args.plot_slac:
        plot_slac_results()

    # Plot data.
    for i in range(len(collect_data)):
        xs, ys = collect_data[i]
        n_experiments = len(xs)

        for exp_i in range(n_experiments):
            xs[exp_i] = xs[exp_i] * info_envs[i]['action_repeat'] # Convert train_step into env_step

        if args.range != -1:
            xs, ys = get_values_with_range(xs, ys, args.range)
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape

        usex = xs[0]
        ymean = np.nanmean(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)

        ystderr = ystd / np.sqrt(len(ys))
        plt.plot(usex, ymean, label='config')
        if args.shaded_err:
            plt.fill_between(usex, ymean - ystderr, ymean + ystderr, alpha=0.4)
        if args.shaded_std:
            plt.fill_between(usex, ymean - ystd, ymean + ystd, alpha=0.2)

    if args.title == '':
        plt.title(plot_titles[0], fontsize='x-large')
    else:
        plt.title(args.title, fontsize='x-large')
    plt.xlabel('Number of env steps', fontsize='x-large')
    plt.ylabel('Episode Return', fontsize='x-large')

    plt.tight_layout()

    legend_name = [directories[i].split('/')[-1] for i in range(len(directories))]
    if args.legend != '':
        assert len(args.legend) == len(
            directories), "Provided legend is not match with number of directories"
        legend_name = args.legend

    if args.plot_drq:
        legend_name.insert(0, 'DrQ')
    if args.plot_slac:
        legend_name.insert(1, 'SLAC')

    plt.legend(legend_name, loc='best', fontsize='x-large')
    plt.show()

def plot_drq_results():
    # Read data from .csv file
    df = pd.read_csv(os.path.join(args.drq_dir, 'data/dmc_planet_bench.csv'))

    # Get data corresponding to environment 'args.env'
    data = df[df['env'] == args.env]

    # Convert data into standard format
    x_concat = data['step'].to_numpy()
    y_concat = data['episode_reward'].to_numpy()
    start_idxs = np.where(x_concat == 0)[0]

    xs, ys = [], []
    for i in range(len(start_idxs)):
        if i == len(start_idxs) - 1:
            start, end = start_idxs[i], -1
        else:
            start, end = start_idxs[i], start_idxs[i + 1]
        xs.append(x_concat[start:end])
        ys.append(y_concat[start:end])

    # Plotting
    if args.range != -1:
        xs, ys = get_values_with_range(xs, ys, args.range)
    xs, ys = pad(xs), pad(ys)
    assert xs.shape == ys.shape

    usex = xs[0]
    ymean = np.nanmean(ys, axis=0)
    ystd = np.nanstd(ys, axis=0)

    ystderr = ystd / np.sqrt(len(ys))
    plt.plot(usex, ymean, label='config')
    if args.shaded_err:
        plt.fill_between(usex, ymean - ystderr, ymean + ystderr, alpha=0.4)
    if args.shaded_std:
        plt.fill_between(usex, ymean - ystd, ymean + ystd, alpha=0.2)

def plot_slac_results():
    def read_df_slac(env_name, expected_num_trials=10):
        fname = os.path.join(args.slac_dir, 'data/slac/dm_control_{}_slac.csv'.format(env_name))
        with open(fname, 'r') as f:
            df = pd.read_csv(f)
        # decoder variance hyperparameter that achieves the best per-task average
        # return across trials averaged over the first half a million environment
        # steps
        df_mean = df[df["OriginalEnvironmentSteps"] < 5e5].groupby(
            "decoder_var").mean()
        best_decoder_var = df_mean.idxmax().get("AverageReturnEvalPolicy")
        df = df[df["decoder_var"] == best_decoder_var]
        assert len(df['trial_id'].unique()) == expected_num_trials
        df['OriginalEnvironmentStepMillions'] = df['OriginalEnvironmentSteps'] / 1e6
        return df
    # Read data from .csv file corresponding to environment 'args.env'
    data = read_df_slac(args.env)

    # Convert data into standard format
    x_concat = data['OriginalEnvironmentSteps'].to_numpy()
    y_concat = data['AverageReturnEvalPolicy'].to_numpy()
    start_idxs = np.where(x_concat == 0)[0]

    xs, ys = [], []
    for i in range(len(start_idxs)):
        if i == len(start_idxs) - 1:
            start, end = start_idxs[i], -1
        else:
            start, end = start_idxs[i], start_idxs[i + 1]
        xs.append(x_concat[start:end])
        ys.append(y_concat[start:end])

    # Plotting
    if args.range != -1:
        xs, ys = get_values_with_range(xs, ys, args.range)
    xs, ys = pad(xs), pad(ys)
    assert xs.shape == ys.shape

    usex = xs[0]
    ymean = np.nanmean(ys, axis=0)
    ystd = np.nanstd(ys, axis=0)

    ystderr = ystd / np.sqrt(len(ys))
    plt.plot(usex, ymean, label='config')
    if args.shaded_err:
        plt.fill_between(usex, ymean - ystderr, ymean + ystderr, alpha=0.4)
    if args.shaded_std:
        plt.fill_between(usex, ymean - ystd, ymean + ystd, alpha=0.2)

if __name__ == '__main__':
    directories = []
    for i in range(len(args.dir)):
        if args.dir[i][-1] == '/':
            directories.append(args.dir[i][:-1])
        else:
            directories.append(args.dir[i])
    plot_multiple_results(directories)
