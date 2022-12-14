import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


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

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

OUT_DIM_108 = {4: 47}

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False,
                 log_image=False):
        super().__init__()

        assert len(obs_shape) == 3

        self.log_image = log_image
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]
        self.projectors = nn.Sequential(
            nn.Linear(num_filters * out_dim * out_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False, detach_mlp=False):
        h = self.forward_conv(obs)

        if detach and not detach_mlp:
            h = h.detach()

        for i in range(len(self.projectors)):
            h = self.projectors[i](h)
            if isinstance(self.projectors[i], nn.Linear):
                self.outputs['fc%s' % (i + 1)] = h
            elif isinstance(self.projectors[i], nn.LayerNorm):
                self.outputs['ln%s' % (i + 1)] = h

        if detach and detach_mlp:
            h = h.detach()
        h_norm = h
        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def copy_projector_weights_from(self, source):
        for i in range(len(self.projectors)):
            if isinstance(self.projectors[i], nn.Linear):
                tie_weights(src=source.projectors[i], trg=self.projectors[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if self.log_image and len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)

        for i in range(len(self.projectors)):
            if isinstance(self.projectors[i], nn.Linear):
                L.log_param('train_encoder/fc%d' % i, self.projectors[i], step)
            elif isinstance(self.projectors[i], nn.LayerNorm):
                L.log_param('train_encoder/ln%d' % i, self.projectors[i], step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )
