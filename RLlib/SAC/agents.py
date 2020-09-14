import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class StateEncoder(nn.Module):
    def __init__(self, dim_state, dim_latent):
        super(StateEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(dim_state, dim_latent), nn.ReLU(),
                                         nn.Linear(dim_latent, dim_latent), nn.ReLU(),
                                         nn.Linear(dim_latent, dim_latent))
        self.ln = nn.LayerNorm(dim_latent)
        self.apply(weights_init_)
                                         
    def forward(self, state, detach=False):
        h = self.encoder(state)
        if detach:
            h = h.detach()
        h_norm = self.ln(h)
        out = torch.tanh(h_norm)
        return out
    
    def copy_conv_weights_from(self, source):
        # share the parameters
        for i, layer in enumerate(source.encoder):
            if isinstance(layer, nn.Linear):
                self.encoder[i].weight = layer.weight
                self.encoder[i].bias = layer.bias


class StateDecoder(nn.Module):
    def __init__(self, dim_latent, dim_state):
        super(StateDecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(dim_latent, dim_latent), nn.ReLU(),
                                     nn.Linear(dim_latent, dim_latent), nn.ReLU(),
                                     nn.Linear(dim_latent, dim_state))
        self.apply(weights_init_)

    def forward(self, h):
        return self.decoder(h)
        

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, dim_latent=None, arch_type='mlp'):
        super(QNetwork, self).__init__()
        self.arch_type = arch_type
        self.dim_latent = dim_latent
        if arch_type == 'rae':
            self.state_encoder = StateEncoder(num_inputs, dim_latent)
            self.num_inputs = dim_latent
        else:
            self.num_inputs = num_inputs

        # Q1 architecture
        self.linear1 = nn.Linear(self.num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(self.num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action, detach=False):
        if self.arch_type == 'rae':
            xu = torch.cat([self.state_encoder(state, detach=detach), action], 1)
        else:
            xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, dim_latent=None, arch_type='mlp'):
        super(GaussianPolicy, self).__init__()
        self.arch_type = arch_type
        self.dim_latent = dim_latent
        if arch_type == 'rae':
            self.state_encoder = StateEncoder(num_inputs, dim_latent)
            self.num_inputs = dim_latent
        else:
            self.num_inputs = num_inputs
        
        self.linear1 = nn.Linear(self.num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)


    def forward(self, state, rnn_state=None, detach=False):
        if self.arch_type == 'rae':
            x = self.state_encoder(state, detach=detach)
        else:
            x = state.clone()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        rnn_state_new = None
        if rnn_state is not None:
            rnn_state_new = self.lstm_cell(x, rnn_state)
            x = rnn_state_new[0].clone()  # fetch hidden states as x
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, rnn_state_new


    def sample(self, state, rnn_state=None, detach=False):
        mean, log_std, rnn_state = self.forward(state, rnn_state, detach=detach)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, rnn_state, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, dim_latent=None, arch_type='mlp'):
        super(DeterministicPolicy, self).__init__()
        self.arch_type = arch_type
        self.dim_latent = dim_latent
        if arch_type == 'rae':
            self.state_encoder = StateEncoder(num_inputs, dim_latent)
            self.num_inputs = dim_latent
        else:
            self.num_inputs = num_inputs
        
        self.linear1 = nn.Linear(self.num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, rnn_state=None, detach=False):
        if self.arch_type == 'rae':
            x = self.state_encoder(state, detach=detach)
        else:
            x = state.clone()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        rnn_state_new = None
        if rnn_state is not None:
            rnn_state_new = self.lstm_cell(x, rnn_state)
            x = rnn_state_new[0].clone()  # fetch hidden states as x
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean, rnn_state_new

    def sample(self, state, rnn_state=None, detach=False):
        mean, rnn_state = self.forward(state, rnn_state, detach=detach)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, rnn_state, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class AttentionPolicy(nn.Module):
    def __init__(self, dim_state_contex, dim_state_atten, dim_action_attention, hidden_dim, mask_size, sal_size):
        super(AttentionPolicy, self).__init__()
        self.mask_size = mask_size
        self.sal_size = sal_size
        assert self.mask_size[0] * self.mask_size[1] == dim_action_attention

        self.linear1 = nn.Linear(dim_state_contex, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim_state_atten)
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 1, 1, stride=1)

        self.apply(weights_init_)

    def forward(self, contex, attention):
        """contex: (B, 64)
        attention: (B, 60)
        """
        # compute attention shift
        x = F.relu(self.linear1(contex))
        x = torch.tanh(self.linear2(x))
        shift = x.view(-1, 1, self.mask_size[0], self.mask_size[1])
        # attention shift
        att = attention.view(-1, 1, self.mask_size[0], self.mask_size[1])
        x = att + shift
        x = F.interpolate(x, self.sal_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, self.mask_size)
        x = torch.sigmoid(self.conv4(x))  # (B, 1, 5, 12)
        x = x.view(-1, self.mask_size[0] * self.mask_size[1])
        return x