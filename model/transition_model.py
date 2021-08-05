
import random
import torch
import torch.nn as nn


LOG_FREQ = 10000

# class DeterministicTransitionModel(nn.Module):
#
#     def __init__(self, encoder_feature_dim, action_shape, layer_width):
#         super().__init__()
#         self.fc = nn. Linear(encoder_feature_dim + action_shape[0], layer_width)
#         # self.fc1 = nn.Linear(layer_width, layer_width)
#         self.ln = nn.LayerNorm(layer_width)
#         self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
#         print("Deterministic transition model chosen.")
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.ln(x)
#         x = torch.relu(x)
#         # x = self.fc1(x)
#         # x = self.ln(x)
#         # x = torch.relu(x)
#
#         mu = self.fc_mu(x)
#         sigma = None
#         return mu, sigma
#
#     def sample_prediction(self, x):
#         mu, sigma = self(x)
#         return mu
#
#
# class ProbabilisticTransitionModel(nn.Module):
#
#     def __init__(self, encoder_feature_dim, action_shape, layer_width, announce=True, max_sigma=1e1, min_sigma=1e-4):
#         super().__init__()
#         self.fc = nn. Linear(encoder_feature_dim + action_shape[0], layer_width)
#         self.ln = nn.LayerNorm(layer_width)
#         self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
#         self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)
#
#         self.max_sigma = max_sigma
#         self.min_sigma = min_sigma
#         assert(self.max_sigma >= self.min_sigma)
#         if announce:
#             print("Probabilistic transition model chosen.")
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.ln(x)
#         x = torch.relu(x)
#
#         mu = self.fc_mu(x)
#         sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
#         sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
#         return mu, sigma
#
#     def sample_prediction(self, x):
#         mu, sigma = self(x)
#         eps = torch.randn_like(sigma)
#         return mu + sigma * eps
#
#
# class EnsembleOfProbabilisticTransitionModels(object):
#
#     def __init__(self, encoder_feature_dim, action_shape, layer_width, ensemble_size=5):
#         self.models = [ProbabilisticTransitionModel(encoder_feature_dim, action_shape, layer_width, announce=False)
#                        for _ in range(ensemble_size)]
#         print("Ensemble of probabilistic transition models chosen.")
#
#     def __call__(self, x):
#         mu_sigma_list = [model.forward(x) for model in self.models]
#         mus, sigmas = zip(*mu_sigma_list)
#         mus, sigmas = torch.stack(mus), torch.stack(sigmas)
#         return mus, sigmas
#
#     def sample_prediction(self, x):
#         model = random.choice(self.models)
#         return model.sample_prediction(x)
#
#     def to(self, device):
#         for model in self.models:
#             model.to(device)
#         return self
#
#     def parameters(self):
#         list_of_parameters = [list(model.parameters()) for model in self.models]
#         parameters = [p for ps in list_of_parameters for p in ps]
#         return parameters


class DeterministicForwardModel(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, action_shape, z_dim, u_dim, hidden_dim,
                 critic, critic_target,
                 arch, use_act_encoder=True, sim_metric='inner', error_weight=1.0):
        super(DeterministicForwardModel, self).__init__()

        assert sim_metric in ['inner', 'bilinear']
        self.sim_metric = sim_metric
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.hidden_dim = hidden_dim
        self.arch = arch
        self.error_weight = error_weight

        if use_act_encoder:
            self.act_emb_dim = u_dim
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
                nn.Linear(z_dim + self.act_emb_dim, z_dim)
            )
            self.error_model = nn.Sequential(
                nn.Linear(z_dim + self.act_emb_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, z_dim)
            )
        else:
            assert 'Not support architecture: ', arch

        if self.sim_metric == 'bilinear':
            self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        else:
            self.W = None
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
            error_model = self.error_weight * self.error_model(z_u_concat)
            z_next = z_next + error_model
        else:
            z_next = z_next
            error_model = None
        return z_next, error_model

    def curvature(self, z, a, delta=0.1, armotized=False):
        z_alias = z.detach().requires_grad_(True)
        a_alias = a.detach().requires_grad_(True)
        eps_z = torch.normal(mean=torch.zeros_like(z), std=torch.empty_like(z).fill_(delta))
        eps_a = torch.normal(mean=torch.zeros_like(a), std=torch.empty_like(a).fill_(delta))

        z_bar = z_alias + eps_z
        a_bar = a_alias + eps_a

        z_bar_next_pred = self.forward_predictor(torch.cat((z_bar, a_bar), dim=1))
        z_alias_next_pred = self.forward_predictor(torch.cat((z_alias, a_alias), dim=1))

        z_dim, a_dim = z.size(1), a.size(1)
        _, B = self.get_jacobian(self.forward_predictor, z_alias, a_alias)
        (grad_z, ) = torch.autograd.grad(z_alias_next_pred, z_alias, grad_outputs=eps_z,
                                         create_graph=True, retain_graph=True)
        grad_u = torch.bmm(B, eps_a.view(-1, a_dim, 1)).squeeze()

        taylor_error = z_bar_next_pred - (grad_z + grad_u) - z_alias_next_pred
        curve_loss = torch.mean(torch.sum(taylor_error.pow(2), dim=1))
        return curve_loss

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
        if self.sim_metric == 'bilinear':
            Wz = torch.matmul(self.W, q.T)  # (z_dim,B)
            logits = torch.matmul(k, Wz)  # (B,B)
        elif self.sim_metric == 'inner':
            logits = torch.matmul(k, q.T)
        else:
            logits = None

        logits = logits - torch.max(logits, 1)[0][:, None]  # subtract max from logits for stability
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


class MarginalDeterministicForwardModel(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, action_shape, z_dim, u_dim, hidden_dim,
                 critic, critic_target,
                 arch, use_act_encoder=True, sim_metric='bilinear', error_weight=1.0):
        super(MarginalDeterministicForwardModel, self).__init__()

        assert sim_metric in ['inner', 'bilinear']
        self.sim_metric = sim_metric
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(z_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, z_dim)


        if self.sim_metric == 'bilinear':
            self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        else:
            self.W = None
        # self.act_encoder.apply(weight_init)
        # self.forward_predictor.apply(weight_init)
        self.outputs = dict()

    def forward(self, z):
        z = self.fc1(z)
        z = self.ln1(z)
        z = torch.relu(z)

        mu = self.fc_mu(z)
        sigma = None
        return mu, sigma

    def compute_logits(self, k, q):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        if self.sim_metric == 'bilinear':
            Wz = torch.matmul(self.W, q.T)  # (z_dim,B)
            logits = torch.matmul(k, Wz)  # (B,B)
        elif self.sim_metric == 'inner':
            logits = torch.matmul(k, q.T)
        else:
            logits = None

        logits = logits - torch.max(logits, 1)[0][:, None]  # subtract max from logits for stability
        return logits

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_forward_model/%s_hist' % k, v, step)


_AVAILABLE_TRANSITION_MODELS = {'': DeterministicForwardModel,
                                'deterministic': DeterministicForwardModel,
                                'probabilistic': None,
                                'ensemble': None,
                                'marginal_deterministic': MarginalDeterministicForwardModel
                                }


def make_transition_model(fdm_type, obs_shape, action_shape, z_dim, u_dim, hidden_dim,
                          critic, critic_target, arch, use_act_encoder=True, sim_metric='bilinear',
                          error_weight=1.0):
    assert fdm_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[fdm_type](
        obs_shape, action_shape, z_dim, u_dim, hidden_dim,
        critic, critic_target, arch, use_act_encoder, sim_metric, error_weight
    )
