import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

import utils
from utils import squash, gaussian_logprob
from encoder import make_encoder, weight_init
from model.transition_model import make_transition_model

LOG_FREQ = 10000


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

    def forward(
            self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False,
            detach_mlp=False
    ):
        obs = self.encoder(obs, detach=detach_encoder, detach_mlp=detach_mlp)

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
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        cnt = 0
        for i in range(len(self.trunk)):
            if isinstance(self.trunk[i], nn.Linear):
                L.log_param('train_actor/fc%d' % cnt, self.trunk[i], step)
                cnt += 1


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

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
    ):
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

    def forward(self, obs, action, detach_encoder=False,
                detach_mlp=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder, detach_mlp=detach_mlp)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        cnt = 0
        for i in range(len(self.Q1.trunk)):
            if isinstance(self.Q1.trunk[i], nn.Linear):
                L.log_param('train_critic/q1_fc%d' % cnt, self.Q1.trunk[i], step)
                L.log_param('train_critic/q2_fc%d' % cnt, self.Q2.trunk[i], step)
                cnt += 1


class SacAuxAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
            self,
            obs_shape, action_shape,
            device,
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3, alpha_beta=0.9,
            actor_lr=1e-3, actor_beta=0.9, actor_log_std_min=-10, actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3, critic_beta=0.9, critic_tau=0.005, critic_target_update_freq=2,
            encoder_type='pixel', encoder_feature_dim=50, num_layers=4, num_filters=32,
            encoder_lr=1e-3, encoder_tau=0.005,
            log_interval=100,
            use_aug=True,
            enc_update_freq=1, fdm_lr=1e-3,
            fdm_arch='linear', sim_metric='bilinear', error_weight=1.0,
            fdm_error_coef=1.0, fdm_pred_coef=1.0, nce_coef=1.0,
            cov_loss_coef=1.0, var_loss_coef=1.0,
            use_act_encoder=False,
            detach_encoder=False,
            detach_mlp=False,
            share_mlp_ac=False,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.encoder_type = encoder_type
        self.use_aug = use_aug

        self.use_act_encoder = use_act_encoder
        self.detach_encoder = detach_encoder
        self.detach_mlp = detach_mlp
        self.u_dim = 50
        self.fdm_hidden_dim = 50
        self.share_mlp_ac = share_mlp_ac
        self.sim_metric = sim_metric

        self.fdm_arch = fdm_arch

        self.f_error_coef = fdm_error_coef
        self.f_pred_coef = fdm_pred_coef
        self.nce_coef = nce_coef
        self.cov_loss_coef = cov_loss_coef
        self.var_loss_coef = var_loss_coef
        self.enc_update_freq = enc_update_freq

        print('[INFO] Use augmentation: ', str(self.use_aug))
        if self.use_aug:
            self.aug_trans = nn.Sequential(kornia.augmentation.RandomCrop((84, 84)))
        else:
            self.aug_trans = nn.Sequential(kornia.augmentation.CenterCrop((84, 84)))

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

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        if self.share_mlp_ac:
            self.actor.encoder.copy_projector_weights_from(self.critic.encoder)

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

        # Forward model scope
        fdm_type = 'deterministic'
        self.forward_model = make_transition_model(fdm_type,
            obs_shape, action_shape, encoder_feature_dim, self.u_dim, self.fdm_hidden_dim,
            self.critic, self.critic_target, self.fdm_arch, use_act_encoder, self.sim_metric,
            error_weight
        )
        self.forward_model = self.forward_model.to(self.device)
        self.forward_model.encoder.copy_conv_weights_from(self.critic.encoder)

        encoder_params = list(self.forward_model.encoder.parameters())
        if self.sim_metric == 'bilinear':
            encoder_params += [self.forward_model.W]

        fdm_params = list(self.forward_model.forward_predictor.parameters())
        if use_act_encoder:
            fdm_params += list(self.forward_model.act_encoder.parameters())
        if self.fdm_arch == 'linear':
            fdm_params += list(self.forward_model.error_model.parameters())

        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=encoder_lr)
        self.forward_optimizer = torch.optim.Adam(fdm_params, lr=fdm_lr)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.forward_model.train(training)

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

    def covariance_loss(self, z):
        N, D = z.shape

        z = z - z.mean(dim=0)
        cov_z = z.T.matmul(z) / (N - 1)

        diag = torch.eye(D, device=z.device)
        cov_loss = cov_z[~diag.bool()].pow_(2).sum() / D
        diag_cov = cov_z[diag.bool()].detach().sum() / D
        return cov_loss, diag_cov

    def variance_loss(self, z):
        eps = 1e-4
        std_z = torch.sqrt(z.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z))
        return std_loss, torch.mean(z.detach().std(dim=0))

    def update_enc_by_aux(self, cur_obs, next_obs, cur_act, reward, L, step):
        with torch.no_grad():
            z_next = self.forward_model.encoder_target(next_obs).detach()

        # TODO: Predict next state
        z_cur = self.forward_model.encoder(cur_obs)
        z_next_pred, error_model = self.forward_model(z_cur, cur_act)

        # TODO: Compute error of next state prediction
        pred_error = F.mse_loss(z_next_pred, z_next)
        fdm_loss = self.f_pred_coef * pred_error
        if error_model is not None:
            reg_error_loss = torch.mean(error_model.pow(2).sum(dim=1))
            fdm_loss += self.f_error_coef * reg_error_loss

        # TODO: Compute standard deviation and covariance of features
        cov_loss, diag_cov = self.covariance_loss(z_cur.detach())
        variance_loss, std_z = self.variance_loss(z_cur.detach())

        # TODO: Compute NCE to contrast positive pair and negative pairs
        queries, keys = z_next_pred, z_next
        logits = self.forward_model.compute_logits(queries, keys)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        nce_loss = self.cross_entropy_loss(logits, labels)

        loss = self.nce_coef * nce_loss + fdm_loss
        # loss = fdm_loss + self.cov_loss_coef * cov_loss + self.var_loss_coef * variance_loss

        self.encoder_optimizer.zero_grad()
        self.forward_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.forward_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/contrastive_loss', nce_loss, step)
            L.log('train_dynamic/enc_cov_loss', cov_loss, step)
            L.log('train_dynamic/enc_diag_cov', diag_cov, step)
            L.log('train_dynamic/enc_std_z', std_z, step)
            L.log('train_dynamic/variance_loss', variance_loss, step)
            L.log('train_dynamic/pred_error', pred_error, step)
            if error_model is not None:
                L.log('train_dynamic/reg_error_loss', reg_error_loss, step)
        self.forward_model.log(L, step)

    def update_encoder(self, obs, next_obs, action, reward, L, step):
        self.update_enc_by_aux(obs, next_obs, action, reward, L, step)

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder,
                                             detach_mlp=self.detach_mlp)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        detach_mlp = True if self.share_mlp_ac else False
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True, detach_mlp=detach_mlp)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True, detach_mlp=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)

        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, _, not_done = replay_buffer.sample()

        obs = self.aug_trans(obs)
        next_obs = self.aug_trans(next_obs)

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.enc_update_freq == 0:
            self.update_encoder(obs, next_obs, action, reward, L, step)

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
