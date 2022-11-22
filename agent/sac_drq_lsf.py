import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import kornia

import math
import utils
from encoder import make_encoder

LOG_FREQ = 10000

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters,):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=False
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, L, step, log_freq=LOG_FREQ):
        return
        # if step % log_freq != 0:
        #     return
        # for k, v in self.outputs.items():
        #     L.log_histogram('train_actor/%s_hist' % k, v, step)
        #
        # L.log_param('train_actor/fc1', self.trunk[0], step)
        # L.log_param('train_actor/fc2', self.trunk[2], step)
        # L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=False
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        return
        # if step % log_freq != 0:
        #     return
        # self.encoder.log(L, step, log_freq)
        #
        # for k, v in self.outputs.items():
        #     L.log_histogram('train_critic/%s_hist' % k, v, step)
        #
        # for i in range(3):
        #     L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
        #     L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class SacLSFAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 hidden_dim=256,
                 discount=0.99,
                 init_temperature=0.01,
                 alpha_lr=1e-3,
                 alpha_beta=0.9,
                 actor_lr=1e-3,
                 actor_beta=0.9,
                 actor_log_std_min=-10,
                 actor_log_std_max=2,
                 actor_update_freq=2,
                 critic_lr=1e-3,
                 critic_beta=0.9,
                 critic_tau=0.005,
                 critic_target_update_freq=2,
                 encoder_type='pixel',
                 encoder_feature_dim=50,
                 encoder_lr=1e-3,
                 decoder_lr=1e-3,
                 encoder_tau=0.005,
                 num_layers=4,
                 num_filters=32,
                 log_interval=100,
                 detach_encoder=False,):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.dynamic_hidden_dim = 256
        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(4),
            kornia.augmentation.RandomCrop((self.image_size, self.image_size))
        )
        self.center_crop = nn.Sequential(
            kornia.augmentation.CenterCrop((self.image_size, self.image_size))
        )

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        # self.reward_decoder = nn.Sequential(
        #     nn.Linear(encoder_feature_dim, self.dynamic_hidden_dim),
        #     nn.LayerNorm(self.dynamic_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.dynamic_hidden_dim, 1),
        # ).to(device)

        self.inverse_model = nn.Sequential(
            nn.Linear(encoder_feature_dim * 2, self.dynamic_hidden_dim),
            nn.LayerNorm(self.dynamic_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.dynamic_hidden_dim, action_shape[0]),
            nn.Tanh()
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        decoder_weight_lambda = 0.0000001
        self.decoder_optimizer = torch.optim.Adam(
            self.inverse_model.parameters(),
            lr=decoder_lr
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        # self.reward_decoder.train(training)
        self.inverse_model.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        # action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return to_np(action[0])

    def select_action(self, obs):
        with torch.no_grad():
            a = self.act(obs, sample=False)
            return a

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            a = self.act(obs, sample=True)
            return a

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step, post_fix='', detach_encoder=False):
        with torch.no_grad():
            # First augmentation
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            # Second augmentation
            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs_aug, next_action_aug)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action, detach_encoder=detach_encoder)
        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

        logger.log('train_critic/loss' + post_fix, critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step, post_fix=''):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss' + post_fix, actor_loss, step)
        logger.log('train_actor/target_entropy' + post_fix, self.target_entropy, step)
        logger.log('train_actor/entropy' + post_fix, -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss' + post_fix, alpha_loss, step)
        logger.log('train_alpha/value' + post_fix, self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_inverse_reward_model(self, obs, action, reward, next_obs, not_done, logger, step,
                                    detach_encoder=False):
        z = self.critic.encoder(obs, detach=detach_encoder)
        z_next = self.critic.encoder(next_obs, detach=detach_encoder)

        z_cat = torch.cat((z, z_next), axis=1)
        pred_act = self.inverse_model(z_cat)
        inv_model_loss = F.mse_loss(pred_act, action)

        # pred_rew = self.reward_decoder(z_next)
        # reward_loss = F.mse_loss(pred_rew, reward)

        # total_loss = inv_model_loss + reward_loss
        total_loss = inv_model_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if step % self.log_interval == 0:
            logger.log('train/idm_loss', inv_model_loss, step)
            # logger.log('train/reward_loss', reward_loss, step)

    def update(self, replay_buffer, logger, step):
        obs_, action, reward, next_obs_, _, not_done, _ = replay_buffer.sample()
        obs = self.aug_trans(obs_)
        obs_aug = self.aug_trans(obs_)
        next_obs = self.aug_trans(next_obs_)
        next_obs_aug = self.aug_trans(next_obs_)

        if step % self.log_interval == 0:
            logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)

        self.update_inverse_reward_model(obs, action, reward, next_obs, not_done, logger, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def update_use_sf(self, replay_buffer, logger, step):
        _, _, _, _, _, _, extra = replay_buffer.sample(only_extra=True)
        sf_obses, sf_next_obses = extra['sf_obses'], extra['sf_next_obses']
        assert sf_obses.shape[2] == 84
        reward = extra['sf_rewards']
        not_done = torch.ones((sf_obses.shape[0], 1), device=self.device).float()

        obs = self.aug_trans(sf_obses)
        obs_aug = self.aug_trans(sf_obses)
        next_obs = self.aug_trans(sf_next_obses)
        next_obs_aug = self.aug_trans(sf_next_obses)

        with torch.no_grad():
            obses_cen =  self.center_crop(sf_obses)
            next_obses_cen = self.center_crop(sf_next_obses)

            z = self.critic.encoder(obses_cen).detach()
            z_next = self.critic.encoder(next_obses_cen).detach()

            z_cat = torch.cat((z, z_next), axis=1)
            action = self.inverse_model(z_cat).detach()


        if step % self.log_interval == 0:
            logger.log('train/batch_reward_lsf', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step, post_fix='_lsf')

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, logger, step, post_fix='_lsf')

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def update_critic_use_sf(self, replay_buffer, logger, step):
        _, _, _, _, _, _, extra = replay_buffer.sample(only_extra=True)
        sf_obses, sf_next_obses = extra['sf_obses'], extra['sf_next_obses']
        assert sf_obses.shape[2] == 84
        reward = extra['sf_rewards']
        not_done = torch.ones((sf_obses.shape[0], 1), device=self.device).float()

        obs = self.aug_trans(sf_obses)
        obs_aug = self.aug_trans(sf_obses)
        next_obs = self.aug_trans(sf_next_obses)
        next_obs_aug = self.aug_trans(sf_next_obses)

        with torch.no_grad():
            obses_cen = self.center_crop(sf_obses)
            next_obses_cen = self.center_crop(sf_next_obses)

            z = self.critic.encoder(obses_cen).detach()
            z_next = self.critic.encoder(next_obses_cen).detach()

            z_cat = torch.cat((z, z_next), axis=1)
            action = self.inverse_model(z_cat).detach()

        if step % self.log_interval == 0:
            logger.log('train/batch_reward_lsf', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step, post_fix='_lsf')

    def update_critic_use_original_data(self, replay_buffer, logger, step):
        obs_, action, reward, next_obs_, _, not_done, _ = replay_buffer.sample()
        obs = self.aug_trans(obs_)
        obs_aug = self.aug_trans(obs_)
        next_obs = self.aug_trans(next_obs_)
        next_obs_aug = self.aug_trans(next_obs_)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)

    def update_dynamics_only(self, replay_buffer, logger, step):
        obs_, action, reward, next_obs_, _, not_done, _ = replay_buffer.sample()
        obs = self.aug_trans(obs_)
        next_obs = self.aug_trans(next_obs_)

        self.update_inverse_reward_model(obs, action, reward, next_obs, not_done, logger, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
