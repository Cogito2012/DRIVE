import sys
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

pi = Variable(torch.FloatTensor([math.pi])).cuda()

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # predict orientation
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, num_outputs, device=torch.device("cuda")):
        self.policy_model = Policy(hidden_size, num_inputs, num_outputs)
        self.device = device
        self.policy_model = self.policy_model.to(device)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=1e-3)
        self.policy_model.train()

    def select_action(self, state):
        """
        state: (128,)
        """
        mu, sigma_sq = self.policy_model(Variable(state))
        sigma_sq = F.softplus(sigma_sq)

        eps = Variable(torch.randn(mu.size())).to(self.device)
        # calculate the probability
        action = (mu + sigma_sq.sqrt()*eps).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).to(self.device)).sum() - (0.0001*entropies[i].to(self.device)).sum()
        loss = loss / len(rewards)
		
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()