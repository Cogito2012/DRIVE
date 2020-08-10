import sys
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

pi = Variable(torch.FloatTensor([math.pi])).cuda()
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # predict orientation
        self.linear_mu = nn.Linear(hidden_size, num_outputs)
        self.linear_sigma = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = self.linear_mu(x)
        sigma_sq = self.linear_sigma(x)
        sigma_sq = torch.clamp(sigma_sq, min=2 * LOG_SIG_MIN, max=2 * LOG_SIG_MAX)

        return mu, sigma_sq


class REINFORCE:
    def __init__(self, num_inputs, dim_action, cfg, device=torch.device("cuda")):
        self.policy_model = Policy(cfg.hidden_size, num_inputs, dim_action)
        self.device = device
        self.policy_model = self.policy_model.to(device)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=1e-3)
        self.num_classes = cfg.num_classes
        # self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.nll_loss = torch.nn.NLLLoss(reduction='none')
        

    def select_action(self, state, with_grad=False):
        """
        state: (1, 128)
        """
        state = Variable(torch.from_numpy(state)).to(self.device)
        mu, sigma_sq = self.policy_model(state)
        sigma_sq = F.softplus(sigma_sq)

        eps = Variable(torch.randn(mu.size())).to(self.device)
        x_t = mu + sigma_sq.sqrt()*eps  # reparameterization trick
        # compute action
        action = torch.tanh(x_t)

        # compute logPI
        prob = normal(x_t.data, mu, sigma_sq)
        log_prob = prob.log()
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # compute entropy
        entropy = -0.5*((sigma_sq + 2*pi.expand_as(sigma_sq)).log() + 1)

        action_out = action if with_grad else action[0].detach().cpu().numpy()
        return action_out, log_prob, entropy


    def update_parameters(self, rewards, log_probs, entropies, states, labels, cfg):
        R = torch.zeros(1, 1)
        policy_loss = 0
        horizon = len(rewards)
        actions = []
        for i in reversed(range(horizon)):
            # compute discounted return
            R = cfg.gamma * R + rewards[i]
            # get the action with grad
            # state = Variable(torch.from_numpy(states[i-horizon+1])).to(self.device)
            state = states[i-horizon+1]
            action, _, _ = self.select_action(state, with_grad=True)
            actions.append(action)
            # total loss
            policy_loss = policy_loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).to(self.device)).sum() - (cfg.alpha * entropies[i]).sum()
        policy_loss = policy_loss / len(rewards)
		
        # compute time-to-accident losses
        task_loss = self.compute_task_loss(actions, labels)
        total_loss = cfg.beta * policy_loss + task_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        utils.clip_grad_norm_(self.policy_model.parameters(), 40)
        self.optimizer.step()

        return total_loss, policy_loss, task_loss


    def compute_task_loss(self, actions, labels):
        """Compute the loss for accident anticipation
        """
        batch_size = labels.shape[0]
        # get prediction
        actions = torch.cat(actions, dim=0)  # T x 3
        score_pred = 0.5 * (actions[:, 2] + 1.0)
        # get label target
        labels = torch.from_numpy(labels).to(self.device)
        steps, clsID, toa, fps = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]
        task_target = torch.zeros(batch_size, self.num_classes).to(self.device)
        task_target.scatter_(1, clsID.unsqueeze(1).long(), 1)  # one-hot
        # compute loss
        task_loss = self._exp_loss(score_pred, task_target, steps / fps, toa)

        return task_loss

    
    def _exp_loss(self, pred, target, time, toa):
        '''
        :param pred:
        :param target: onehot codings for binary classification
        :param time:
        :param toa:
        :return:
        '''
        pred = torch.cat([(1.0 - pred).unsqueeze(1), pred.unsqueeze(1)], dim=1)
        # positive example (exp_loss)
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), toa.to(pred.dtype) - time - 1)
        penalty = torch.where(toa > 0, penalty, torch.zeros_like(penalty).to(pred.device))

        # pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        pos_loss = -torch.mul(torch.exp(penalty), -self.nll_loss(torch.log(pred + 1e-6), target_cls))
        # negative example
        # neg_loss = self.ce_loss(pred, target_cls)
        neg_loss = self.nll_loss(torch.log(pred + 1e-6), target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss


    def load_models(self, ckpt_dir, args):
        filename = sorted(os.listdir(ckpt_dir))[-1]
        weight_file = os.path.join(args.output, 'checkpoints', filename)
        if os.path.isfile(weight_file):
            checkpoint = torch.load(weight_file)
            self.policy_model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(weight_file))
        else:
            raise FileNotFoundError