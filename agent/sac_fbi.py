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

    def __init__(self, obs_shape, action_shape, z_dim, a_emb_dim, critic, critic_target,
                 arch, use_act_encoder=True):
        super(ForwardModel, self).__init__()

        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.hidden_dim = 50
        self.arch = arch

        if use_act_encoder:
            self.act_emb_dim = 50
            self.act_encoder = nn.Sequential(
                nn.Linear(action_shape[0], self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.act_emb_dim)
            )
        else:
            self.act_emb_dim = action_shape[0]
            self.act_encoder = None

        print('[INFO] Forward model architecture: ', arch)
        if arch == 'non_linear':
            self.forward_predictor = nn.Sequential(
                nn.Linear(z_dim + self.act_emb_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, z_dim),
            )
            self.error_model = None
        elif arch == 'linear':
            self.forward_predictor = nn.Sequential(
                nn.Linear(z_dim + self.act_emb_dim, self.hidden_dim)
            )
            self.error_model = nn.Sequential(
                nn.Linear(z_dim + self.act_emb_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, z_dim)
            )
        else:
            assert 'Not support architecture: ', arch

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        # self.act_encoder.apply(weight_init)
        # self.forward_predictor.apply(weight_init)
        self.outputs = dict()

    def forward(self, z, a):
        if self.act_encoder is not None:
            u = self.act_encoder(a)
        else:
            u = a
        z_u_concat = torch.cat((z, u), dim=1)
        z_next = self.forward_predictor(z_u_concat)
        if self.arch == 'linear':
            error_model = self.error_model(z_u_concat)
            z_next = z_next + error_model
        else:
            z_next = z_next
            error_model = None
        return z_next, error_model

    def curvature(self, z, u, delta=0.1, armotized=False):
        z_alias = z.detach().requires_grad_(True)
        u_alias = u.detach().requires_grad_(True)
        eps_z = torch.normal(mean=torch.zeros_like(z), std=torch.empty_like(z).fill_(delta))
        eps_u = torch.normal(mean=torch.zeros_like(u), std=torch.empty_like(u).fill_(delta))

        z_bar = z_alias + eps_z
        u_bar = u_alias + eps_u

        z_bar_next_pred = self.forward_predictor(torch.cat((z_bar, u_bar), dim=1))
        z_alias_next_pred = self.forward_predictor(torch.cat((z_alias, u_alias), dim=1))

        z_dim, u_dim = z.size(1), u.size(1)
        _, B = self.get_jacobian(self.forward_predictor, z_alias, u_alias)
        (grad_z, ) = torch.autograd.grad(z_alias_next_pred, z_alias, grad_outputs=eps_z,
                                         create_graph=True, retain_graph=True)
        grad_u = torch.bmm(B, eps_u.view(-1, u_dim, 1)).squeeze()

        taylor_error = z_bar_next_pred - (grad_z + grad_u) - z_alias_next_pred
        cur_loss = torch.mean(torch.sum(taylor_error.pow(2), dim=1))
        return cur_loss

    @staticmethod
    def get_jacobian(dynamics, batched_z, batched_u):
        """
        compute the jacobian of F(z,a) w.r.t z, a
        """
        batch_size = batched_z.size(0)
        z_dim = batched_z.size(-1)

        z, u = batched_z.unsqueeze(1), batched_u.unsqueeze(1)  # batch_size, 1, input_dim
        z, u = z.repeat(1, z_dim, 1), u.repeat(1, z_dim, 1)  # batch_size, output_dim, input_dim
        z_next = dynamics(torch.cat((z, u), dim=2))
        grad_inp = torch.eye(z_dim).reshape(1, z_dim, z_dim).repeat(batch_size, 1, 1).cuda()
        all_A, all_B = torch.autograd.grad(z_next, [z, u], grad_outputs=[grad_inp, grad_inp],
                                           create_graph=True, retain_graph=True)
        return all_A, all_B

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
        if self.act_encoder is not None:
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


def lerp(global_step, start_step, end_step, start_val, end_val):
    assert end_step > start_step, "end_step must be larger start_step: %d <= %d" % \
                                  (end_step, start_step)
    interp = (global_step - start_step) / (end_step - start_step)
    interp = np.maximum(0.0, np.minimum(1.0, interp))
    weight = start_val * (1.0 - interp) + end_val * interp
    return weight


def subexpd_np(step, start_d, start_v, d_decay_rate, v_decay_rate, start_t=0,
               stair=False):
    step -= start_t
    exp = np.log(-np.log(d_decay_rate) * step / start_d + 1) / -np.log(d_decay_rate)
    if stair:
        exp = np.floor(exp)
    return start_v * v_decay_rate ** exp


def subexpd(global_step, start_step, end_step, start_val, end_val,
            warmup=True, start_t=0, stair=True):
  """Sub-exponential decay function. Duration decay is sqrt(decay)."""
  if warmup and start_step == 0:
    return lerp(global_step, start_step, end_step, start_val, end_val)
  decay_steps = end_step - start_step
  decay_factor = end_val
  d_decay_factor = np.sqrt(decay_factor)
  step = global_step - start_step
  return subexpd_np(step, decay_steps, start_val, d_decay_factor, decay_factor, start_t)


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
            fdm_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            fdm_update_freq=1,
            log_interval=100,
            detach_encoder=False,
            no_aug=False,
            target_entropy='dimA',
            use_reg=False,
            enc_fw_e2e=False,
            fdm_arch='linear',
            fdm_error_coef=1.0,
            use_act_encoder=True,
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
        self.use_reg = use_reg

        self.use_act_encoder = use_act_encoder


        # self.pi_arch = 'linear'
        # self.q_arch = 'linear'
        self.fdm_arch = fdm_arch
        self.pi_arch = 'non_linear'
        self.q_arch = 'non_linear'
        # self.fdm_arch = 'non_linear'
        self.enc_fw_e2e = enc_fw_e2e

        self.fdm_error_coef = fdm_error_coef
        self.fdm_update_freq = fdm_update_freq

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
        if target_entropy == 'dimA':
            self.target_entropy = -np.prod(action_shape)
        elif target_entropy == 'dimA2':
            self.target_entropy = -np.prod(action_shape)/2
        else:
            self.target_entropy = None

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
        a_emb_dim = 50
        self.forward_model = ForwardModel(
            obs_shape, action_shape, encoder_feature_dim, a_emb_dim,
            self.critic, self.critic_target, self.fdm_arch, use_act_encoder
        ).to(self.device)

        encoder_params = list(self.forward_model.encoder.parameters()) + [self.forward_model.W]
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

    def update_fw_alt(self, cur_obs, next_obs, cur_act, reward, L, step):
        # TODO: Train FDM alternatively
        assert not self.enc_fw_e2e
        # Step 1: Freeze "obs encoder", learn "act encoder & forward model & error model"
        with torch.no_grad():
            z_cur = self.forward_model.encoder(cur_obs).detach()
            z_next = self.forward_model.encoder_target(next_obs).detach()

        # s' = Ws*s + Wa*a + error(s,a)
        z_next_pred, error_model = self.forward_model(z_cur, cur_act)

        pred_loss = F.mse_loss(z_next_pred, z_next)
        if self.fdm_arch == 'linear':
            reg_error_loss = 0.5 * torch.norm(error_model, p=2) ** 2
            fdm_loss = pred_loss + self.fdm_error_coef * reg_error_loss
        else:
            reg_error_loss = None
            fdm_loss = pred_loss

        self.forward_optimizer.zero_grad()
        fdm_loss.backward()
        self.forward_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/pred_loss', pred_loss, step)
            if self.fdm_arch == 'linear':
                L.log('train_dynamic/error_model', error_model.abs().mean(), step)
                L.log('train_dynamic/reg_error_loss', reg_error_loss, step)

        # Step 2: Freeze "act encoder & forward & error model", learn obs encoder
        z_cur = self.forward_model.encoder(cur_obs)
        z_next_pred, _ = self.forward_model(z_cur, cur_act)

        # reg_error_loss = 0.5 * torch.norm(error_model, p=2) ** 2
        queries, keys = z_next_pred, z_next

        logits = self.forward_model.compute_logits(queries, keys)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        nce_loss = self.cross_entropy_loss(logits, labels)

        loss = nce_loss

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/contrastive_loss', nce_loss, step)
        self.forward_model.log(L, step)

    def update_fw_e2e(self, cur_obs, next_obs, cur_act, reward, L, step):
        # TODO: Train FDM e2e
        assert self.enc_fw_e2e
        z_cur = self.forward_model.encoder(cur_obs)
        with torch.no_grad():
            z_next = self.forward_model.encoder_target(next_obs).detach()

        # s' = Ws*s + Wa*a + error(s,a)
        z_next_pred, error_model = self.forward_model(z_cur, cur_act)

        # pred_loss = F.mse_loss(z_next_pred, z_next)
        # reg_error_loss = 0.5 * torch.norm(error_model, p=2) ** 2
        # fdm_loss = pred_loss + self.fdm_error_coef * reg_error_loss
        # fdm_loss = pred_loss

        queries, keys = z_next_pred, z_next
        logits = self.forward_model.compute_logits(queries, keys)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        nce_loss = self.cross_entropy_loss(logits, labels)

        loss = nce_loss
        # loss = nce_loss + fdm_loss

        self.encoder_optimizer.zero_grad()
        self.forward_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.forward_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/contrastive_loss', nce_loss, step)
            if self.fdm_arch == 'linear':
                L.log('train_dynamic/error_model', error_model.abs().mean(), step)
                # L.log('train_dynamic/pred_loss', fdm_loss, step)
                # L.log('train_dynamic/reg_error_loss', reg_error_loss, step)
        self.forward_model.log(L, step)

    def update_fw_e2e_curvature(self, cur_obs, next_obs, cur_act, reward, L, step):
        cur_coef = 1.0
        assert self.enc_fw_e2e and self.fdm_arch == 'non_linear'
        z_cur = self.forward_model.encoder(cur_obs)
        a_cur = self.forward_model.act_encoder(cur_act)
        with torch.no_grad():
            z_next_gt = self.forward_model.encoder_target(next_obs).detach()

        # s' = Ws*s + Wa*a + error(s,a)
        z_next_pred = self.forward_model.forward_predictor(torch.cat((z_cur, a_cur), dim=1))
        # if self.fdm_arch == 'linear':
        #     error_model = self.forward_model.error_model(torch.cat((z_cur, a_cur), dim=1))
        #     z_next_pred = z_next_pred + error_model
        # else:
        #     error_model = None

        queries, keys = z_next_pred, z_next_gt
        logits = self.forward_model.compute_logits(queries, keys)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)

        cur_loss = self.forward_model.curvature(z_cur, a_cur)

        loss = contrastive_loss + cur_coef * cur_loss

        self.encoder_optimizer.zero_grad()
        self.forward_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.forward_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_dynamic/contrastive_loss', contrastive_loss, step)
            L.log('train_dynamic/cur_loss', cur_loss, step)
            # if self.fdm_arch == 'linear':
            #     L.log('train_dynamic/error_model', error_model.abs().mean(), step)
        self.forward_model.log(L, step)

    def update_encoder(self, obs, next_obs, action, reward, L, step):
        if self.enc_fw_e2e:
            self.update_fw_e2e(obs, next_obs, action, reward, L, step)
            # self.update_fw_e2e_curvature(obs, next_obs, action, reward, L, step)
        else:
            self.update_fw_alt(obs, next_obs, action, reward, L, step)

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

    def update_critic_drq(self, obs, obs_aug, action, reward,
                          next_obs, next_obs_aug, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

            _, policy_action_aug, log_pi_aug, _ = self.actor(next_obs_aug)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug, policy_action_aug)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action, detach_encoder=self.detach_encoder)
        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

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

    def update(self, replay_buffer, L, step):
        if self.use_reg:
            obs_aug, action, reward, next_obs_aug, not_done, drq_kwargs = replay_buffer.sample_drq()
            obs, next_obs = drq_kwargs['obses_origin'], drq_kwargs['next_obses_origin']
        else:
            obs, action, reward, next_obs, not_done, _ = replay_buffer.sample_cpc(self.no_aug)

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        if self.use_reg:
            self.update_critic_drq(obs, obs_aug, action, reward,
                                   next_obs, next_obs_aug, not_done, L, step)
        else:
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

        if step % self.fdm_update_freq == 0:
            self.update_encoder(obs, next_obs, action, reward, L, step)


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
