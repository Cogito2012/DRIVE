import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

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


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        # predict orientation
        self.linear_mu = nn.Linear(hidden_size, num_outputs)
        self.linear_sigma = nn.Linear(hidden_size, num_outputs)
        self.apply(weights_init_)  # initialize params

    def forward(self, inputs, rnn_state=None):
        x = inputs
        x = F.relu(self.linear1(x))
        rnn_state_new = None
        if rnn_state is not None:
            rnn_state_new = self.lstm_cell(x, rnn_state)
            x = rnn_state_new[0].clone()  # fetch hidden states as x

        mu = self.linear_mu(x)
        sigma_sq = self.linear_sigma(x)
        sigma_sq = torch.clamp(sigma_sq, min=2 * LOG_SIG_MIN, max=2 * LOG_SIG_MAX)

        return mu, sigma_sq, rnn_state_new


class ValueEstimator(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(ValueEstimator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.apply(weights_init_)  # initialize params

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        output = self.linear2(x)
        return output