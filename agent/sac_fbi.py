import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder

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



def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters,
            arch
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        print ('[INFO] Actor architecture: ', arch)
        if arch == 'non_linear':
            self.trunk = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2 * action_shape[0])
            )
        elif arch == 'linear':
            self.trunk = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, 2 * action_shape[0])
            )
        else:
            assert 'Architecture not support: ', arch

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
            self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
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

    def __init__(self, obs_dim, action_dim, hidden_dim, arch):
        super().__init__()

        if arch == 'non_linear':
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        elif arch == 'linear':
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, 1)
            )
        else:
            assert 'Architecture not support: ', arch

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, arch
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        print('[INFO] Critic_1 architecture: ', arch)
        print('[INFO] Critic_2 architecture: ', arch)
        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim, arch
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim, arch
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


class ForwardModel(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, action_shape, z_dim, critic, critic_target, hidden_dim,
                 encoder_type, encoder_feature_dim, num_layers, num_filters,
                 arch):
        super(ForwardModel, self).__init__()

        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.hidden_dim = 50

        self.act_encoder = nn.Sequential(
            nn.Linear(action_shape[0], self.encoder.feature_dim), nn.ReLU(),
            nn.Linear(self.encoder.feature_dim, self.encoder.feature_dim)
        )

        print('[INFO] Forward model architecture: ', arch)
        if arch == 'non_linear':
            self.forward_predictor = nn.Sequential(
                nn.Linear(self.encoder.feature_dim * 2, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.error_model = None
        elif arch == 'linear':
            self.forward_predictor = nn.Sequential(
                nn.Linear(self.encoder.feature_dim * 2, self.hidden_dim)
            )
            self.error_model = nn.Sequential(
                nn.Linear(self.encoder.feature_dim * 2, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            assert 'Not support architecture: ', arch

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        # self.act_encoder.apply(weight_init)
        # self.forward_predictor.apply(weight_init)
        self.outputs = dict()

    def encode(self, x, detach_encoder=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach_encoder:
            z_out = z_out.detach()
        return z_out

    def predict_next(self, o, a, detach_encoder=False):
        z_o_cur = self.encode(o, ema=False, detach_encoder=detach_encoder)
        z_a_cur = self.act_encoder(a)
        z_o_next = self.forward_predictor(torch.cat((z_o_cur, z_a_cur), dim=1))
        return z_o_next


    def compute_logits(self, k, q):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, q.T)  # (z_dim,B)
        logits = torch.matmul(k, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_forward_model/%s_hist' % k, v, step)

        cnt = 0
        for i in range(len(self.forward_predictor)):
            if isinstance(self.forward_predictor[i], nn.Linear):
                L.log_param('train_forward_model/fdm%d' % cnt, self.forward_predictor[i], step)
                cnt += 1
        cnt = 0
        for i in range(len(self.act_encoder)):
            if isinstance(self.act_encoder[i], nn.Linear):
                L.log_param('train_forward_model/act_emb%d' % cnt, self.act_encoder[i], step)
                cnt += 1

        if self.error_model is not None:
            cnt = 0
            for i in range(len(self.error_model)):
                if isinstance(self.error_model[i], nn.Linear):
                    L.log_param('train_forward_model/err_model%d' % cnt, self.error_model[i], step)
                    cnt += 1


class SacFbiAgent(object):
    """CURL representation learning with SAC."""

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
            idm_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            fdm_update_freq=1,
            bdm_update_freq=np.inf,
            idm_update_freq=np.inf,
            log_interval=100,
            detach_encoder=False,
            no_aug=False,
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
        self.no_aug = no_aug

        # self.pi_arch = 'linear'
        # self.q_arch = 'linear'
        self.fdm_arch = 'linear'
        self.pi_arch = 'non_linear'
        self.q_arch = 'non_linear'
        # self.fdm_arch = 'non_linear'
        self.enc_fw_e2e = False

        self.fdm_error_coef = 1

        self.fdm_update_freq = fdm_update_freq
        self.bdm_update_freq = bdm_update_freq
        self.idm_update_freq = idm_update_freq

        print('[INFO] Use augmentation: ', str(not self.no_aug))

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, self.pi_arch
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, self.q_arch
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, self.q_arch
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

        self.forward = ForwardModel(
            obs_shape, action_shape, encoder_feature_dim, self.critic, self.critic_target, hidden_dim,
            encoder_type, encoder_feature_dim, num_layers, num_filters,
            self.fdm_arch,
        ).to(self.device)

        if self.enc_fw_e2e:
            self.forward_optimizer = torch.optim.Adam(
                self.forward.parameters(), lr=encoder_lr
            )
        else:
            if self.fdm_arch == 'linear':
                self.encoder_optimizer = torch.optim.Adam(
                    list(self.forward.encoder.parameters()) +
                    [self.forward.W], lr=encoder_lr)

                self.forward_optimizer = torch.optim.Adam(
                    list(self.forward.forward_predictor.parameters()) +
                    list(self.forward.act_encoder.parameters()) +
                    list(self.forward.error_model.parameters()), lr=encoder_lr
                )
            elif self.fdm_arch == 'non_linear':
                self.encoder_optimizer = torch.optim.Adam(
                    list(self.forward.encoder.parameters()) +
                    [self.forward.W], lr=encoder_lr
                )
                self.forward_optimizer = torch.optim.Adam(
                    list(self.forward.forward_predictor.parameters()) +
                    list(self.forward.act_encoder.parameters()), lr=encoder_lr
                )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.forward.train(training)

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

    def update_forward_v1(self, cur_obs, next_obs, cur_act, L, step):
        # TODO: For non-linear FDM, train alternatively
        # Step 1: Freeze obs encoder, learn act encoder, forward & error model
        with torch.no_grad():
            z_cur = self.forward.encoder(cur_obs).detach()
            z_next_gt = self.forward.encoder_target(next_obs).detach()

        a_cur = self.forward.act_encoder(cur_act)
        # s' = f(s,a)
        z_next_pred = self.forward.forward_predictor(torch.cat((z_cur, a_cur), dim=1))

        pred_loss = F.mse_loss(z_next_pred, z_next_gt)
        fdm_loss = pred_loss

        self.forward_optimizer.zero_grad()
        fdm_loss.backward()
        self.forward_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/pred_loss', pred_loss, step)

        # Step 2: Freeze act encoder, forward & error model, learn obs encoder
        z_cur = self.forward.encoder(cur_obs)
        a_cur = self.forward.act_encoder(cur_act)

        z_next_pred = self.forward.forward_predictor(torch.cat((z_cur, a_cur), dim=1))

        queries, keys = z_next_pred, z_next_gt

        logits = self.forward.compute_logits(queries, keys)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/contrastive_loss', loss, step)
        self.forward.log(L, step)

    def update_forward_v1_1(self, cur_obs, next_obs, cur_act, L, step):
        # TODO: For non-linear FDM, train E2E
        z_cur = self.forward.encoder(cur_obs)
        a_cur = self.forward.act_encoder(cur_act)
        with torch.no_grad():
            z_next_gt = self.forward.encoder_target(next_obs).detach()

        # s' = f(s,a)
        z_next_pred = self.forward.forward_predictor(torch.cat((z_cur, a_cur), dim=1))

        queries, keys = z_next_pred, z_next_gt

        logits = self.forward.compute_logits(queries, keys)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)

        loss = contrastive_loss

        self.forward_optimizer.zero_grad()
        loss.backward()
        self.forward_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/contrastive_loss', loss, step)
        self.forward.log(L, step)

    def update_forward_v2(self, cur_obs, next_obs, cur_act, L, step):
        # TODO: For linear FDM, train alternatively
        # Step 1: Freeze obs encoder, learn act encoder, forward & error model
        with torch.no_grad():
            z_cur = self.forward.encoder(cur_obs).detach()
            z_next_gt = self.forward.encoder_target(next_obs).detach()

        a_cur = self.forward.act_encoder(cur_act)
        # s' = Ws*s + Wa*a + error(s,a)
        z_next_pred_linear = self.forward.forward_predictor(torch.cat((z_cur, a_cur), dim=1))
        error_model = self.forward.error_model(torch.cat((z_cur, a_cur), dim=1))
        z_next_pred = z_next_pred_linear + error_model

        pred_loss = F.mse_loss(z_next_pred, z_next_gt)
        reg_error_loss = 0.5 * torch.norm(error_model, p=2) ** 2
        fdm_loss = pred_loss + self.fdm_error_coef * reg_error_loss

        self.forward_optimizer.zero_grad()
        fdm_loss.backward()
        self.forward_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/pred_loss', pred_loss, step)
            L.log('train_dynamic/error_model', error_model.abs().mean(), step)
            L.log('train_dynamic/reg_error_loss', reg_error_loss, step)

        # Step 2: Freeze act encoder, forward & error model, learn obs encoder
        z_cur = self.forward.encoder(cur_obs)
        a_cur = self.forward.act_encoder(cur_act)

        z_next_pred_linear = self.forward.forward_predictor(torch.cat((z_cur, a_cur), dim=1))
        error_model = self.forward.error_model(torch.cat((z_cur, a_cur), dim=1))

        z_next_pred = z_next_pred_linear + error_model
        queries, keys = z_next_pred, z_next_gt

        logits = self.forward.compute_logits(queries, keys)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/contrastive_loss', loss, step)
        self.forward.log(L, step)

    def update_forward_v2_1(self, cur_obs, next_obs, cur_act, L, step):
        # TODO: For linear FDM, train E2E
        z_cur = self.forward.encoder(cur_obs)
        a_cur = self.forward.act_encoder(cur_act)
        with torch.no_grad():
            z_next_gt = self.forward.encoder_target(next_obs).detach()

        # s' = Ws*s + Wa*a + error(s,a)
        z_next_pred_linear = self.forward.forward_predictor(torch.cat((z_cur, a_cur), dim=1))
        error_model = self.forward.error_model(torch.cat((z_cur, a_cur), dim=1))

        z_next_pred = z_next_pred_linear + error_model

        # reg_error_loss = 0.5 * torch.norm(error_model, p=2) ** 2

        queries, keys = z_next_pred, z_next_gt
        logits = self.forward.compute_logits(queries, keys)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)

        # loss = contrastive_loss + self.fdm_error_coef * reg_error_loss
        loss = contrastive_loss

        self.forward_optimizer.zero_grad()
        loss.backward()
        self.forward_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/contrastive_loss', contrastive_loss, step)
            L.log('train_dynamic/error_model', error_model.abs().mean(), step)
            # L.log('train_dynamic/reg_error_loss', reg_error_loss, step)
        self.forward.log(L, step)

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder)
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
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

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

    def update(self, replay_buffer, L, step, train_rl=True):
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, _ = replay_buffer.sample_cpc(no_aug=self.no_aug)
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        if train_rl:
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

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if step % self.fdm_update_freq == 0:
            # print('[INFO] Training with: Forward model')
            if self.fdm_arch == 'non_linear':
                if not self.enc_fw_e2e:
                    self.update_forward_v1(obs, next_obs, action, L, step)
                else:
                    self.update_forward_v1_1(obs, next_obs, action, L, step)
            elif self.fdm_arch == 'linear':
                if not self.enc_fw_e2e:
                    self.update_forward_v2(obs, next_obs, action, L, step)
                else:
                    self.update_forward_v2_1(obs, next_obs, action, L, step)
            else:
                assert 'Error'

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
