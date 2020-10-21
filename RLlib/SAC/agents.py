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


class AccidentPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, dim_latent=None, arch_type='mlp', policy_type='Gaussian'):
        super(AccidentPolicy, self).__init__()
        self.arch_type = arch_type
        self.policy_type = policy_type
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
        if self.policy_type == 'Gaussian':
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        else:
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
        mean = self.mean_linear(x)
        if self.policy_type == 'Gaussian':
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            log_std = torch.tensor(0.0).to(mean.device)
        return mean, log_std, rnn_state_new


    def sample(self, state, rnn_state=None, detach=False):
        mean, log_std, rnn_state = self.forward(state, rnn_state, detach=detach)
        if self.policy_type == 'Gaussian':
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
        else:
            noise = self.noise.normal_(0., std=0.1)
            noise = noise.clamp(-0.25, 0.25)
            action = mean + noise
            log_prob = torch.tensor(0.0).to(action.device)
        return action, rnn_state, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        if not self.policy_type == 'Gaussian':
            self.noise = self.noise.to(device)
        return super(AccidentPolicy, self).to(device)


class FixationPolicy(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_dim, dim_latent=None, arch_type='mlp', policy_type='Gaussian'):
        super(FixationPolicy, self).__init__()
        self.arch_type = arch_type
        self.policy_type = policy_type
        self.dim_latent = dim_latent
        if arch_type == 'rae':
            self.state_encoder = StateEncoder(dim_state, dim_latent)
            self.num_inputs = dim_latent
        else:
            self.num_inputs = dim_state

        self.linear1 = nn.Linear(self.num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, dim_action)
        if self.policy_type == 'Gaussian':
            self.log_std_linear = nn.Linear(hidden_dim, dim_action)
        else:
            self.noise = torch.Tensor(dim_action)
        self.apply(weights_init_)

    def forward(self, state, detach=False):
        """
        state: (B, 64)
        """
        if self.arch_type == 'rae':
            x = self.state_encoder(state, detach=detach)
        else:
            x = state.clone()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        if self.policy_type == 'Gaussian':
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            mean = torch.tanh(mean)
            log_std = torch.tensor(0.0).to(mean.device)
        return mean, log_std

    def sample(self, state, detach=False):
        mean, log_std = self.forward(state, detach=detach)
        if self.policy_type == 'Gaussian':
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t.clone()
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(1 - y_t.pow(2) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean)
        else:
            noise = self.noise.normal_(0., std=0.1)
            noise = noise.clamp(-0.25, 0.25)
            action = mean + noise
            log_prob = torch.tensor(0.0).to(action.device)
        return action, log_prob, mean
        

    def to(self, device):
        if not self.policy_type == 'Gaussian':
            self.noise = self.noise.to(device)
        return super(FixationPolicy, self).to(device)