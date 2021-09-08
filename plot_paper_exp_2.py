import argparse
import ast
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob2
import matplotlib.ticker as ticker

from baselines.common import plot_util
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

from plot_dmc import get_data_in_subdir, pad, get_values_with_range


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, nargs='+')
parser.add_argument('--radius', type=int, default=0)
parser.add_argument('--range', type=int, default=-1, help='Number of transitions want to plot')
parser.add_argument('--legend', type=str, default='', nargs='+')
parser.add_argument('--ylim', type=float, default=0)
parser.add_argument('--title', type=str, default='')
parser.add_argument('--shaded_std', type=bool, default=True)
parser.add_argument('--shaded_err', type=bool, default=False)
parser.add_argument('--train_test', action='store_true')

parser.add_argument('--env', type=str, default='cheetah_run')

parser.add_argument('--save_path', type=str, default='/home/tung/workspace/rlbench/sac_baselines/figures')

parser.add_argument('--ar', type=int, nargs='+', default=None, help='Multiple w/ action repeat')
args = parser.parse_args()


def plot_multiple_results(directories):
    # color_table = ['k', '#ff7c00', '#e8000b', '#1ac938', '#9f4800', '#8b2be2', '#023eff', '#f14cc1','#a3a3a3', '#ffc400', '#00d7ff']
    # Color:        den,   cam,       do,        xanh la,    tim,       brown      xanh nuoc bien
    # color_table = ['#1ac938', '#ff7c00', '#e8000b', '#9f4800', '#8b2be2', '#023eff', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff']
    # Color:        xanh la,   cam,       do,         tim,       brown      xanh nuoc bien
    color_table = ['#1ac938', '#e8000b', '#023eff', '#ff7c00']
    linestyle = ['-', '-', '-', '-']

    # User config:
    x_key = 'step'
    y_key = 'mean_episode_reward'

    rc = {'axes.facecolor': 'white',
          'legend.fontsize': 12,
          'axes.titlesize': 15,
          'axes.labelsize': 15,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'axes.formatter.useoffset': False,
          'axes.formatter.offset_threshold': 1}

    plt.rcParams.update(rc)

    fig, ax = plt.subplots()

    collect_data, plot_titles, info_envs = [], [], []
    for directory in directories:
        data_in_subdir, task_name, info_env = get_data_in_subdir(directory, x_key, y_key)
        collect_data.append(data_in_subdir)
        plot_titles.append(task_name)
        info_envs.append(info_env)

    # Plot data.
    for i in range(len(collect_data)):
        xs, ys = collect_data[i]
        n_experiments = len(xs)

        if args.env == 'manipulation' and i == 0:
            filter_idxes = np.arange(51) * 4
        else:
            filter_idxes = None

        if filter_idxes is not None:
            filtered_xs, filter_ys = [], []
            for exp_i in range(n_experiments):
                _xs = xs[exp_i][filter_idxes]
                filtered_xs.append(_xs)
                filter_ys.append(ys[exp_i][filter_idxes])
            xs, ys = filtered_xs, filter_ys

        for exp_i in range(n_experiments):
            xs[exp_i] = xs[exp_i] * info_envs[i]['action_repeat']

        if args.range != -1:
            xs, ys = get_values_with_range(xs, ys, args.range)
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape

        usex = xs[0]
        ymean = np.nanmean(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)

        linewidth = 2.5 if i == 2 else 1.5
        plt.plot(usex, ymean, color=color_table[i], linestyle=linestyle[i], linewidth=linewidth)
        if args.shaded_std:
            plt.fill_between(usex, ymean - ystd, ymean + ystd, alpha=0.1, color=color_table[i])

    # plt.grid(True, which='major', color='grey', linestyle='--')

    # if args.legend != '':
    #     assert len(args.legend) == len(directories), "Provided legend is not match with number of directories"
    #     legend_name = args.legend
    # else:
    #     legend_name = [directories[i].split('/')[-1] for i in range(len(directories))]

    # legend_name = ["Vanilla SAC", "SAC+avg.", "SAC+ours"]
    legend_name = ["SAC", "SAC+IDM", "SAC+ours"]

    plt.legend(legend_name, loc='best', fontsize='x-large')
    # plt.legend(legend_name, loc='lower right', frameon=True,
    #            facecolor='#f2f2f2', edgecolor='grey')
    # plt.legend(legend_name, loc='lower right', frameon=True)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    env_titles = dict(
        finger_spin='Finger-spin',
        cartpole_swingup='Cartpole-swingup',
        reacher_easy='Reacher-easy',
        cheetah_run='Cheetah-run',
        walker_walk='Walker-walk',
        ball_in_cup_catch='Ball_in_cup-catch',
        manipulation='Reach-Duplo'
    )
    plot_xlabels = dict(
        cartpole_swingup=r'Environment steps ($\times 1e6$)',
        cheetah_run=r'Environment steps ($\times 1e6$)',
        ball_in_cup_catch=r'Environment steps ($\times 1e6$)',
    )
    plt.title(env_titles[args.env])
    plt.xlabel(plot_xlabels[args.env])
    plt.ylabel('Episode Return')

    env_lims = dict(
        cartpole_swingup=[[0, 500000], [1, 800]],
        ball_in_cup_catch=[[0, 500000], [1, 950]],
        cheetah_run=[[0, 500000], [1, 600]],
        manipulation=[[0, 400000], [1, 230]],
    )
    plt.xlim(env_lims[args.env][0][0], env_lims[args.env][0][1])
    plt.ylim(env_lims[args.env][1][0], env_lims[args.env][1][1])

    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e6)))
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, 'ab_1_{}.pdf'.format(args.env)))
    plt.show()


if __name__ == '__main__':
    directory = []

    experiments = dict(
        cartpole_swingup=[
            'logs/cartpole-swingup/sac_lsf_rad_1000init_1gs_aclr1e-4_clr1e-4_allr1e-4_elr1e-4_ac_freq2_2_enccri_tau0.05_0.01_declr1e-4_bs128_baseline_extra_upd_critic_uselsf_0_extraupd_1_n_invupd_1_aug_0_alpha0.5/',
            'logs/cartpole-swingup/sac_lsf_rad_1000init_1gs_aclr1e-4_clr1e-4_allr1e-4_elr1e-4_ac_freq2_2_enccri_tau0.05_0.01_declr1e-4_bs128_baseline_extra_upd_critic_uselsf_0_extraupd_1_n_invupd_1_aug_0_alpha0.5_idm_aux/',
            'logs/cartpole-swingup/sac_lsf_rad_1000init_1gs_aclr1e-3_clr1e-3_allr1e-4_elr1e-3_ac_freq2_2_enccri_tau0.05_0.01_declr1e-3_bs128_baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0_alpha0.5'
        ],
        ball_in_cup_catch=[
            's159_logs/ball_in_cup-catch/sac_pixel/',
            's159_logs/ball_in_cup-catch/sac_lsf_rad_1000init_1gs_aclr1e-4_clr1e-4_allr1e-4_elr1e-4_ac_freq2_2_enccri_tau0.05_0.01_declr1e-4_bs128_baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0_alpha0.5_idm_aux/',
            's159_logs/ball_in_cup-catch/sac_lsf_rad_1000init_1gs_aclr1e-4_clr1e-4_allr1e-4_elr1e-4_ac_freq2_2_enccri_tau0.05_0.01_declr1e-4_bs128_baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0_alpha0.5/'
        ],
        cheetah_run=[
            'logs/cheetah-run/sac_lsf_rad_1000init_1gs_aclr1e-4_clr1e-4_allr1e-4_elr1e-4_ac_freq2_2_enccri_tau0.05_0.01_declr1e-4_bs128_baseline_extra_upd_critic_uselsf_0_extraupd_1_n_invupd_1_aug_0/',
            'logs/cheetah-run/sac_lsf_rad_1000init_1gs_aclr1e-4_clr1e-4_allr1e-4_elr1e-4_ac_freq2_2_enccri_tau0.05_0.01_declr1e-4_bs128_baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0_idm_aux/',
            'logs/cheetah-run/sac_lsf_rad_1000init_1gs_aclr1e-3_clr1e-3_allr1e-4_elr1e-3_ac_freq2_2_enccri_tau0.05_0.01_declr1e-3_bs128_baseline_extra_upd_critic_uselsf_1_extraupd_1_n_invupd_1_aug_0/'
        ],
    )
    env_ranges = dict(
        cartpole_swingup=1000000,
        ball_in_cup_catch=1000000,
        cheetah_run=1000000,
        manipulation=-1
    )
    # for i in range(len(args.dir)):
    #     if args.dir[i][-1] == '/':
    #         directory.append(args.dir[i][:-1])
    #     else:
    #         directory.append(args.dir[i])

    # envs = ['finger_spin', 'cartpole_swingup', 'reacher_easy',
    #         'ball_in_cup_catch']
    envs = [args.env]
    # Override param
    for env in envs:
        args.range = env_ranges[env]
    for env in envs:
        folder = experiments[env]
        plot_multiple_results(folder)