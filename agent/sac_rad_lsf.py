import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

import utils
from encoder import make_encoder
import transformation.data_augs as rad

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


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


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
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

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        return
        # if step % log_freq != 0:
        #     return
        #
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
    """Critic network, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
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
        # detach_encoder allows to stop gradient propogation to encoder
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
        #
        # self.encoder.log(L, step, log_freq)
        #
        # for k, v in self.outputs.items():
        #     L.log_histogram('train_critic/%s_hist' % k, v, step)
        #
        # for i in range(3):
        #     L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
        #     L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class SacRadLSFAgent(object):
    """RAD with SAC."""

    def __init__(
            self,
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
            detach_encoder=False,
            batch_size=None,
            action_repeat=1,
            use_aug=True
    ):
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
        self.batch_size = batch_size
        self.action_repeat = action_repeat

        self.augs_funcs = {}

        self.dynamic_hidden_dim = 256
        if use_aug:
            self.aug_trans = nn.Sequential(
                kornia.augmentation.RandomCrop((self.image_size, self.image_size))
            )
        else:
            self.aug_trans = nn.Sequential(
                kornia.augmentation.CenterCrop((self.image_size, self.image_size))
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
        self.inverse_model.apply(weight_init)

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

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step,
                      post_fix='', detach_encoder=False):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)

            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if step % self.log_interval == 0:
            L.log('train_critic/loss' + post_fix, critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step, post_fix=''):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss' + post_fix, actor_loss, step)
            L.log('train_actor/target_entropy' + post_fix, self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
                  (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy' + post_fix, entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss' + post_fix, alpha_loss, step)
            L.log('train_alpha/value' + post_fix, self.alpha, step)
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

    def update(self, replay_buffer, L, step, use_lsf=False, n_inv_updates=1):
        obs, action, reward, next_obs, _, not_done, _ = replay_buffer.sample()
        assert obs.shape[2] == 100

        obs = self.aug_trans(obs)
        next_obs = self.aug_trans(next_obs)

        if use_lsf:
            for i in range(n_inv_updates):
                if i > 0:
                    obs, action, reward, next_obs, _, not_done, _ = replay_buffer.sample()

                    obs = self.aug_trans(obs)
                    next_obs = self.aug_trans(next_obs)
                self.update_inverse_reward_model(obs, action, reward, next_obs, not_done, L, step)

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

    def update_v1(self, replay_buffer, L, step, use_lsf=False, n_repeats=3):
        obs, action, reward, next_obs, _, not_done, _ = replay_buffer.sample(batch_size=self.batch_size)

        if use_lsf:
            _, _, _, _, _, _, extra = replay_buffer.sample(only_extra=True,
                                                           batch_size=int(self.batch_size * (n_repeats - 1)))
            sf_obses, sf_next_obses, sf_reward = extra['sf_obses'], extra['sf_next_obses'], extra['sf_rewards']
            assert sf_obses.shape[2] == 100
            sf_not_done = torch.ones((sf_obses.shape[0], 1), device=self.device).float()

            with torch.no_grad():
                obses_cen = self.center_crop(sf_obses)
                next_obses_cen = self.center_crop(sf_next_obses)

                z = self.critic.encoder(obses_cen)
                z_next = self.critic.encoder(next_obses_cen)

                z_cat = torch.cat((z, z_next), axis=1)
                sf_action = self.inverse_model(z_cat).detach()
            obs_ex, action_ex, reward_ex, next_obs_ex, not_done_ex = \
                sf_obses, sf_action, sf_reward, sf_next_obses, sf_not_done
        else:
            obs_ex, action_ex, reward_ex, next_obs_ex, _, not_done_ex, _ = replay_buffer.sample(
                batch_size=int(self.batch_size * (n_repeats - 1)))

        # Data for Q-training
        obs_q = torch.cat((obs, obs_ex))
        next_obs_q = torch.cat((next_obs, next_obs_ex))
        reward_q = torch.cat((reward, reward_ex))
        action_q = torch.cat((action, action_ex))
        not_done_q = torch.cat((not_done, not_done_ex))

        assert obs.shape[2] == 100
        obs_q = self.aug_trans(obs_q)
        next_obs_q = self.aug_trans(next_obs_q)
        obs = obs_q[:self.batch_size, :, :, :]
        next_obs = next_obs_q[:self.batch_size, :, :, :]

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        if use_lsf:
            self.update_inverse_reward_model(obs, action, reward, next_obs, not_done, L, step)

        # self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_critic(obs_q, action_q, reward_q, next_obs_q, not_done_q, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

    def update_use_sf(self, replay_buffer, L, step):
        _, _, _, _, _, _, extra = replay_buffer.sample(only_extra=True)
        sf_obses, sf_next_obses, sf_reward = extra['sf_obses'], extra['sf_next_obses'], extra['sf_rewards']
        assert sf_obses.shape[2] == 100
        not_done = torch.ones((sf_obses.shape[0], 1), device=self.device).float()
        reward = sf_reward

        obs = self.aug_trans(sf_obses)
        next_obs = self.aug_trans(sf_next_obses)

        with torch.no_grad():
            obses_cen =  self.center_crop(sf_obses)
            next_obses_cen = self.center_crop(sf_next_obses)

            z = self.critic.encoder(obses_cen)
            z_next = self.critic.encoder(next_obses_cen)

            z_cat = torch.cat((z, z_next), axis=1)
            action = self.inverse_model(z_cat).detach()

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step,
                           post_fix='_lsf')

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, post_fix='_lsf')

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

    def update_critic_use_sf(self, replay_buffer, L, step):
        _, _, _, _, _, _, extra = replay_buffer.sample(only_extra=True)
        sf_obses, sf_next_obses, sf_reward = extra['sf_obses'], extra['sf_next_obses'], extra['sf_rewards']
        assert sf_obses.shape[2] == 100
        not_done = torch.ones((sf_obses.shape[0], 1), device=self.device).float()
        reward = sf_reward

        obs = self.aug_trans(sf_obses)
        next_obs = self.aug_trans(sf_next_obses)

        with torch.no_grad():
            obses_cen =  self.center_crop(sf_obses)
            next_obses_cen = self.center_crop(sf_next_obses)

            z = self.critic.encoder(obses_cen)
            z_next = self.critic.encoder(next_obses_cen)

            z_cat = torch.cat((z, z_next), axis=1)
            action = self.inverse_model(z_cat).detach()

        if step % self.log_interval == 0:
            L.log('train/batch_reward_lsf', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

    def update_critic_use_sf_previous_method(self, replay_buffer, L, step):
        _, _, _, _, _, _, extra = replay_buffer.sample(only_extra=True)
        sf_obses, sf_next_obses, sf_reward, sf_action = \
            extra['sf_obses'], extra['sf_next_obses'], extra['sf_rewards'], extra['sf_actions']
        assert sf_obses.shape[2] == 100
        not_done = torch.ones((sf_obses.shape[0], 1), device=self.device).float()
        reward = sf_reward
        action = sf_action

        obs = self.aug_trans(sf_obses)
        next_obs = self.aug_trans(sf_next_obses)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

    def update_critic_use_original_data(self, replay_buffer, L, step):
        obs, action, reward, next_obs, _, not_done, _ = replay_buffer.sample()
        assert obs.shape[2] == 100

        obs = self.aug_trans(obs)
        next_obs = self.aug_trans(next_obs)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

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
