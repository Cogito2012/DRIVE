# import sys
import math
import os

import torch.__config__
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable

from .agents import Policy, ValueEstimator
from src.data_transform import scales_to_point, norm_fix
from metrics.losses import exp_loss, fixation_loss

class REINFORCE(object):
    def __init__(self, cfg, device=torch.device("cuda")):
        self.policy_model = Policy(cfg.hidden_size, cfg.dim_state, cfg.dim_action).to(device)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=cfg.lr)
        self.device = device
        self.num_classes = cfg.num_classes
        self.pi = Variable(torch.FloatTensor([math.pi])).to(device)
        # whether to use baseline
        self.with_baseline = cfg.with_baseline
        if self.with_baseline:
            self.value_estimator = ValueEstimator(cfg.hidden_size, cfg.dim_state, 1).to(device)
            self.advantage_loss = torch.nn.MSELoss(reduction='none')
            self.value_optimizer = optim.Adam(self.value_estimator.parameters(), lr=cfg.lr_adv)

    def normal(self, x, mu, sigma_sq):
        a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
        b = 1/(2*sigma_sq*self.pi.expand_as(sigma_sq)).sqrt()
        return a*b

    def set_status(self, phase='train'):
        isTraining = True if phase == 'train' else False
        self.policy_model.train(isTraining)
        if self.with_baseline:
            self.value_estimator.train(isTraining)

    def select_action(self, state, rnn_state=None, with_grad=False):
        """
        state: (B, 128)
        """
        if rnn_state is not None:
            rnn_state = (rnn_state[0], rnn_state[1])

        mu, sigma_sq, rnn_state = self.policy_model(state, rnn_state)
        sigma_sq = F.softplus(sigma_sq)

        eps = Variable(torch.randn(mu.size())).to(self.device)
        x_t = mu + sigma_sq.sqrt()*eps  # reparameterization trick
        # compute action
        action = torch.tanh(x_t)

        # compute logPI
        prob = self.normal(x_t.data, mu, sigma_sq)
        log_prob = prob.log()
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # compute entropy
        entropy = -0.5*((sigma_sq + 2*self.pi.expand_as(sigma_sq)).log() + 1)

        action_out = action if with_grad else action.detach()
        return action_out, log_prob, entropy, rnn_state


    def update_parameters(self, rewards, log_probs, entropies, states, rnn_state, all_times, all_fixations, env, cfg):
        R = torch.zeros(1, 1).to(self.device)
        policy_loss, adv_loss = 0, 0
        horizon = len(rewards)
        actions = []
        for i in reversed(range(horizon)):
            # compute discounted return
            R = cfg.REINFORCE.gamma * R + rewards[i]
            if self.with_baseline:
                # compute the baseline
                baseline = self.value_estimator(Variable(states[i]))
                adv_loss = adv_loss + torch.mean(self.advantage_loss(baseline, Variable(R)))
                R = R - baseline.detach()  # advantage
            # total loss
            policy_loss = policy_loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() - (cfg.REINFORCE.alpha * entropies[i]).sum()

            # get the action with grad
            state = states[i-horizon+1]
            action, _, _, rnn_state = self.select_action(state, rnn_state, with_grad=True)  # (B, 1, 3)
            actions.append(action.unsqueeze(1))
        actions = torch.cat(actions, dim=1)  # B x T x 3

        policy_loss = policy_loss / len(rewards)
        adv_loss = adv_loss / len(rewards)

        if self.with_baseline:
            # update value estimator
            self.value_optimizer.zero_grad()
            adv_loss.backward()
            utils.clip_grad_norm_(self.value_estimator.parameters(), 40)
            self.value_optimizer.step()
		
        # compute time-to-accident losses
        cls_loss, fix_loss = self.compute_task_loss(actions, all_times, all_fixations, env, cfg.ENV)
        total_loss = policy_loss + cfg.REINFORCE.beta_accident * cls_loss + cfg.REINFORCE.beta_fixation * fix_loss
        # update policy model
        self.optimizer.zero_grad()
        total_loss.backward()
        utils.clip_grad_norm_(self.policy_model.parameters(), 40)
        self.optimizer.step()

        # gather losses
        losses = {'total': total_loss.item(), 'policy/actor': policy_loss.item(), 'policy/accident': cls_loss.item(), 'policy/fixation': fix_loss.item()}
        if self.with_baseline:
            losses.update({'advantage': adv_loss.item()})

        return losses


    def compute_task_loss(self, actions, all_times, all_fixations, env, cfg):
        """Compute the loss for accident anticipation
        actions: (B, T, 3)
        all_times: (B, T)
        all_fixations: (B, T, 2)
        """
        batch_size, horizon = actions.size(0), actions.size(1)
        # get prediction
        score_pred = 0.5 * (actions[:, :, 0] + 1.0)  # map to [0, 1], shape=(B, T)
        fix_pred = actions[:, :, 1:].view(-1, 2) # (B*T, 2)  (x,y)

        # compute the early anticipation loss
        toa_batch = env.begin_accident.unsqueeze(1).repeat(1, horizon).view(-1)
        clsID_batch = env.clsID.unsqueeze(1).repeat(1, horizon).view(-1)  # (B*T,)
        cls_target = torch.zeros(batch_size * horizon, self.num_classes).to(self.device)
        cls_target.scatter_(1, clsID_batch.unsqueeze(1).long(), 1)  # one-hot
        cls_loss = exp_loss(score_pred.view(-1), cls_target, all_times.view(-1), toa_batch)  # expoential binary cross entropy

        # fixation loss
        fix_batch = all_fixations.view(-1, 2)  # (B*T, 2)
        mask = fix_batch[:, 0].bool().float() * fix_batch[:, 1].bool().float()  # (B,)
        fix_gt = fix_batch[mask.bool()]
        fix_pred = fix_pred[mask.bool()]
        fix_pred = scales_to_point(fix_pred, cfg.image_shape, cfg.input_shape)  # scaling scales to point
        fix_loss = torch.sum(torch.pow(norm_fix(fix_pred, cfg.input_shape) - norm_fix(fix_gt, cfg.input_shape), 2), dim=1).mean()  # (B) [0, sqrt(2)]

        return cls_loss, fix_loss


    def load_models(self, ckpt_dir, cfg):
        if cfg.test_epoch == -1:
            filename = sorted(os.listdir(ckpt_dir))[-1]
            weight_file = os.path.join(cfg.output, 'checkpoints', filename)
        else:
            weight_file = os.path.join(cfg.output, 'checkpoints', 'reinforce_epoch_%02d.pth'%(cfg.test_epoch))
        if os.path.isfile(weight_file):
            checkpoint = torch.load(weight_file, map_location=self.device)
            self.policy_model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(weight_file))
        else:
            raise FileNotFoundError