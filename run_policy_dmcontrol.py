import numpy as np
import torch
import argparse
import os
import time
import json
import dmc2gym
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils
from utils import str2bool
from train import make_agent, make_env
from video import VideoRecorder


def parse_args():
    parser = argparse.ArgumentParser()
    ## pre-trained encoder
    parser.add_argument('--dir', default='.', type=str)
    parser.add_argument('--step', default=None, type=int)
    parser.add_argument('--n_tests', default=10, type=int)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--logdir', type=str, default='.')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--pause', type=float, default=0.001)

    parser.add_argument('--benchmark', default='planet', type=str, choices=['dreamer', 'planet'])
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=-1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # Distractor environment
    parser.add_argument('--difficulty', default='easy', type=str,
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--bg_dataset_path',
                        default='/home/tung/workspace/rlbench/DAVIS/JPEGImages/480p/', type=str)
    parser.add_argument('--bg_dynamic', action='store_true')
    parser.add_argument('--rand_bg', action='store_true')
    parser.add_argument('--rand_cam', action='store_true')
    parser.add_argument('--rand_color', action='store_true')

    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    # RAD
    parser.add_argument('--data_augs', default='crop', type=str)
    # CURL
    parser.add_argument('--cpc_update_freq', default=1, type=int)
    # CPM
    parser.add_argument('--enc_update_freq', default=1, type=int)
    parser.add_argument('--bdm_update_freq', default=1, type=int)
    parser.add_argument('--idm_update_freq', default=1, type=int)
    parser.add_argument('--cpm_noaug', action='store_true', default=False)
    # Leveraging skipped frames (LSF)
    parser.add_argument('--n_extra_update_cri', default=1, type=int)
    parser.add_argument('--use_lsf', type=str2bool, default=False)
    parser.add_argument('--use_aug', type=str2bool, default=True)
    parser.add_argument('--n_inv_updates', default=2, type=int)
    # Linearized FDM
    parser.add_argument('--fdm_lr', default=1e-3, type=float)
    parser.add_argument('--fdm_arch', default='linear', type=str)
    parser.add_argument('--sim_metric', default='bilinear', type=str, choices=['bilinear', 'inner'])

    parser.add_argument('--fdm_error_coef', default=1.0, type=float)
    parser.add_argument('--fdm_pred_coef', default=1.0, type=float)
    parser.add_argument('--nce_coef', default=1.0, type=float)

    parser.add_argument('--enc_fw_e2e', type=str2bool, default=True)
    parser.add_argument('--use_act_encoder', type=str2bool, default=False)
    parser.add_argument('--detach_encoder', type=str2bool, default=False)
    parser.add_argument('--detach_mlp', type=str2bool, default=False)
    parser.add_argument('--share_mlp_ac', type=str2bool, default=False)
    parser.add_argument('--use_rew_pred', type=str2bool, default=False)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='sac_fbi', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--num_train_envsteps', default=-1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--n_grad_updates', default=1, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--eval_seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--exp', default='exp', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args


def get_args_from_checkpoint(path):
    json_file = os.path.join(path, 'args.json')
    with open(json_file) as f:
        ckpt_args = json.load(f)
    return ckpt_args


def update_args(src_args_dict, des_args):
    if des_args.seed is None:
        args.seed = np.random.randint(1, 1000000)
        print('[INFO] Seed: ', args.seed)

    if args.step is None:
        all_ckpt_actors = glob.glob(os.path.join(des_args.dir, 'model', 'actor_*'))
        ckpt_step = []
        for ckpt in all_ckpt_actors:
            ckpt_step.append(int(ckpt.split('/')[-1].split('.')[0].split('_')[-1]))
        args.step = max(ckpt_step)
    assert isinstance(src_args_dict, dict), 'src_args_dict must be dictionary for paring'
    exclude_args = ['seed', 'dir', 'n_tests', 'step']
    for arg in src_args_dict.keys():
        if arg in exclude_args or arg not in des_args.__dict__.keys():
            continue
        des_args.__dict__[arg] = src_args_dict[arg]
    return des_args

def imshow(obs, pause=0.001):
    plt.imshow(obs)
    plt.axis('off')
    plt.tight_layout()
    plt.pause(pause)
    plt.ion()
    plt.show()
    # plt.show(block=False)

def render(obs, stack_frame=1, pause=0.001):
    img = obs[:3, :, :].transpose(1, 2, 0)
    imshow(img, pause)

def preprocess_obs(obs):
    if args.agent in ['sac_curl', 'sac_cpm', 'sac_rad_lsf']:
        preprocessed = utils.center_crop_image(obs, args.image_size)  # Preprocess input for CURL
    elif args.agent in ['sac_rad']:
        # center crop image
        if 'crop' in args.data_augs:
            obs = utils.center_crop_image(obs, args.image_size)
        if 'translate' in args.data_augs:
            # first crop the center with pre_image_size
            obs = utils.center_crop_image(obs, args.pre_transform_image_size)
            # then translate cropped to center
            obs = utils.center_translate(obs, args.image_size)
        preprocessed = obs
    else:
        preprocessed = obs
    return preprocessed


"""
Example to run: for rendering
python collect_demonstrations.py --dir ./logs/../pixel-rgb-crop-s1-2020_10_12_22_46_25 --step 0 --render
or, for recording video:
python run_policy_dm_control.py --dir ./logs/../pixel-rgb-crop-s1-2020_10_12_22_46_25 --step 0 --n_tests 1 --logdir . --record --cpu

where, step is checkpoint step, if `step` is not provided, it takes the last step.
"""
def main(args):
    ckpt_args = get_args_from_checkpoint(args.dir)
    args = update_args(ckpt_args, args)

    utils.set_seed_everywhere(args.seed)

    env = make_env(args, model='train')
    eval_env_seed = dict(
        cheetah=[2, 10]
    )
    if args.domain_name in eval_env_seed.keys():
        env_seed = int(np.random.choice(eval_env_seed[args.domain_name]))
    else:
        env_seed = args.seed
    args.__dict__["env_seed"] = env_seed
    env.seed(env_seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack, ar=args.action_repeat)

    if not args.cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    if args.encoder_type == 'pixel':
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
    elif args.encoder_type == 'identity':
        obs_shape = env.observation_space.shape
    else:
        obs_shape = None
        assert 'Encoder is not supported: %s' % args.encoder_type

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    if args.record:
        video_dir = args.logdir
        video = VideoRecorder(video_dir if args.record else None)
    else:
        video = None

    model_dir = os.path.join(args.dir, 'model')
    agent.load(model_dir=model_dir, step=args.step)

    n_tests = args.n_tests

    all_ep_rewards = []
    start_time = time.time()

    print("\n\n******** TESTING ********")
    print("[TEST-INFO] Domain-task: %s-%s" % (args.domain_name, args.task_name))
    print("[TEST-INFO] Seed: %d" % (args.seed))
    print("[TEST-INFO] Env seed: %d" % (args.env_seed))
    print("[TEST-INFO] # tests: %d" % (n_tests))
    print("[TEST-INFO] Checkpoint at env step: %d" % (args.step * args.action_repeat))

    for i in tqdm(range(n_tests)):
        obs = env.reset()
        if args.record:
            video.init(enabled=True)
            video.record(env)
        if args.render:
            render(obs, pause=args.pause)
        done = False
        episode_reward = 0
        while not done:
            obs = preprocess_obs(obs)
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            if args.render:
                render(obs, pause=args.pause)
            if args.record:
                video.record(env)
            episode_reward += reward

        all_ep_rewards.append(episode_reward)
        if args.record:
            video.save('{}-{}_step{}_trial{}.mp4'.format(
                args.domain_name, args.task_name, args.step * args.action_repeat, i + 1))
    print("eval/eval_time: %.4f (s)" % (time.time() - start_time))

    mean_ep_reward = np.mean(all_ep_rewards)
    std_ep_reward = np.std(all_ep_rewards)
    best_ep_reward = np.max(all_ep_rewards)
    print("eval/episode_reward: mean=%.4f/std=%.4f" % (mean_ep_reward, std_ep_reward))
    print("eval/best_episode_reward: %.4f" % best_ep_reward)



if __name__ == '__main__':
    args = parse_args()
    main(args)