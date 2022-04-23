import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from .utils import soft_update, hard_update
from .agents import AccidentPolicy, FixationPolicy, QNetwork, StateDecoder
from src.data_transform import scales_to_point, norm_fix
from metrics.losses import exp_loss, fixation_loss
import time


class SAC(object):
    def __init__(self, cfg, device=torch.device("cuda")):
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.beta_accident = cfg.beta_accident
        self.beta_fixation = cfg.beta_fixation
        self.losses = {}

        self.arch_type = cfg.arch_type
        self.type_acc = cfg.type_acc
        self.type_fix = cfg.type_fix
        self.actor_update_interval = cfg.actor_update_interval
        self.target_update_interval = cfg.target_update_interval
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning
        self.num_classes = cfg.num_classes
        self.device = device
        self.batch_size = cfg.batch_size
        self.image_size = cfg.image_shape
        self.input_size = cfg.input_shape
        self.pure_sl = cfg.pure_sl if hasattr(cfg, 'pure_sl') else False

        # state dims
        self.dim_state = cfg.dim_state
        self.dim_state_acc = int(0.5 * cfg.dim_state)
        self.dim_state_fix = int(0.5 * cfg.dim_state)
        # action dims
        self.dim_action_acc = cfg.dim_action_acc
        self.dim_action_fix = cfg.dim_action_fix
        self.dim_action = cfg.dim_action_acc + cfg.dim_action_fix

        # create actor and critics
        self.policy_accident, self.policy_fixation, self.critic, self.critic_target = self.create_actor_critics(cfg)
        hard_update(self.critic_target, self.critic)

        # optimizers
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.lr)
        self.policy_acc_optim = Adam(self.policy_accident.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.policy_att_optim = Adam(self.policy_fixation.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        if cfg.arch_type == 'rae':
            # the encoder is shared between two actors and critic, weights need to be tied
            self.policy_accident.state_encoder.copy_conv_weights_from(self.critic.state_encoder)
            self.policy_fixation.state_encoder.copy_conv_weights_from(self.critic.state_encoder)
            # decoder
            self.decoder = StateDecoder(cfg.dim_latent, self.dim_state).to(device=self.device)
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optim = Adam(self.critic.state_encoder.parameters(), lr=cfg.lr)
            # optimizer for decoder
            self.decoder_optim = Adam(self.decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.latent_lambda = cfg.latent_lambda
        
        if self.automatic_entropy_tuning:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.type_acc == "Gaussian" and not self.type_fix == 'Gaussian':
                dim_entropy = self.dim_action_acc
            elif not self.type_acc == "Gaussian" and self.type_fix == 'Gaussian':
                dim_entropy = self.dim_action_fix
            elif self.type_acc == "Gaussian" and self.type_fix == 'Gaussian':
                dim_entropy = self.dim_action
            else:
                print("When automatic entropy, at least one policy is Gaussian!")
                raise ValueError
            self.target_entropy = - dim_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr_alpha)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False


    def create_actor_critics(self, cfg):
        # create critic networks
        critic = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(device=self.device)
        critic_target = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(self.device)
        # create accident anticipation policy
        dim_state = self.dim_state if cfg.arch_type == 'rae' else self.dim_state_acc
        policy_accident = AccidentPolicy(dim_state, self.dim_action_acc, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_acc).to(self.device)
        # create fixation prediction policy
        dim_state = self.dim_state if cfg.arch_type == 'rae' else self.dim_state_fix
        policy_fixation = FixationPolicy(dim_state, self.dim_action_fix, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_fix).to(self.device)
        return policy_accident, policy_fixation, critic, critic_target

    
    def set_status(self, phase='train'):
        isTraining = True if phase == 'train' else False
        self.policy_accident.train(isTraining)
        self.policy_fixation.train(isTraining)
        self.critic.train(isTraining)
        self.critic_target.train(isTraining)
        if self.arch_type == 'rae':
            self.decoder.train(isTraining) 


    def select_action(self, state, rnn_state=None, evaluate=False):
        """state: (B, 64+64), [state_max, state_avg]
        """
        state_max = state[:, :self.dim_state_acc]
        state_avg = state[:, self.dim_state_acc:]
        acc_state = state.clone() if self.arch_type == 'rae' else state_max
        fix_state = state.clone() if self.arch_type == 'rae' else state_avg
        # execute actions
        if evaluate is False:
            action_acc, rnn_state, _, _ = self.policy_accident.sample(acc_state, rnn_state)
            action_fix, _, _ = self.policy_fixation.sample(fix_state)
        else:
            _, rnn_state, _, action_acc = self.policy_accident.sample(acc_state, rnn_state)
            _, _, action_fix = self.policy_fixation.sample(fix_state)
        # get actions
        actions = torch.cat([action_acc.detach(), action_fix.detach()], dim=1)  # (B, 3)
        if rnn_state is not None:
            rnn_state = (rnn_state[0].detach(), rnn_state[1].detach())
        return actions, rnn_state


    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, rnn_state_batch):
        with torch.no_grad():
            # split the next_states
            next_state_max = next_state_batch[:, :self.dim_state_acc]
            next_state_avg = next_state_batch[:, self.dim_state_acc:]
            next_acc_state = next_state_batch.clone() if self.arch_type == 'rae' else next_state_max
            next_fix_state = next_state_batch.clone() if self.arch_type == 'rae' else next_state_avg
            # inference two policies
            next_acc_state_action, _, next_acc_state_log_pi, _ = self.policy_accident.sample(next_acc_state, rnn_state_batch)
            next_fix_state_action, next_fix_state_log_pi, _ = self.policy_fixation.sample(next_fix_state)
            next_state_action = torch.cat([next_acc_state_action, next_fix_state_action], dim=1)
            # interence critics
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * (next_acc_state_log_pi + next_fix_state_log_pi)
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        self.losses.update({'critic': qf_loss.item()})


    def update_actor(self, state_batch, rnn_state_batch, labels_batch):
        # split the states
        state_max = state_batch[:, :self.dim_state_acc]
        state_avg = state_batch[:, self.dim_state_acc:]
        acc_state = state_batch.clone() if self.arch_type == 'rae' else state_max
        fix_state = state_batch.clone() if self.arch_type == 'rae' else state_avg

        # sampling
        pi_acc, _, log_pi_acc, mean_acc = self.policy_accident.sample(acc_state, rnn_state_batch, detach=True)
        pi_fix, log_pi_fix, mean_fix = self.policy_fixation.sample(fix_state, detach=True)
        pi = torch.cat([pi_acc, pi_fix], dim=1)
        log_pi = log_pi_acc + log_pi_fix

        qf1_pi, qf2_pi = self.critic(state_batch, pi, detach=True)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # actor loss
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # actor_loss = - min_qf_pi.mean()

        # compute the early anticipation loss
        score_pred = 0.5 * (mean_acc + 1.0).squeeze(1)  # (B,)
        curtime_batch, clsID_batch, toa_batch, fix_batch = labels_batch[:, 0], labels_batch[:, 1], labels_batch[:, 2], labels_batch[:, 3:5]
        cls_target = torch.zeros(score_pred.size(0), self.num_classes).to(self.device)
        cls_target.scatter_(1, clsID_batch.unsqueeze(1).long(), 1)  # one-hot
        cls_loss = exp_loss(score_pred, cls_target, curtime_batch, toa_batch)  # expoential binary cross entropy

        # fixation loss
        mask = fix_batch[:, 0].bool().float() * fix_batch[:, 1].bool().float()  # (B,)
        fix_gt = fix_batch[mask.bool()]
        fix_pred = mean_fix[mask.bool()]
        fix_pred = scales_to_point(fix_pred, self.image_size, self.input_size)  # scaling scales to point
        fix_loss = torch.sum(torch.pow(norm_fix(fix_pred, self.input_size) - norm_fix(fix_gt, self.input_size), 2), dim=1).mean()  # (B) [0, sqrt(2)]

        if self.pure_sl:
            # for pure supervised learning, we just discard the losses from reinforcement learning
            acc_policy_loss = self.beta_accident * cls_loss
            fix_policy_loss = self.beta_fixation * fix_loss
        else:
            # weighted sum 
            acc_policy_loss = actor_loss.detach() + self.beta_accident * cls_loss
            fix_policy_loss = actor_loss.detach() + self.beta_fixation * fix_loss
        # update accident predictor
        self.policy_acc_optim.zero_grad()
        acc_policy_loss.backward()
        self.policy_acc_optim.step()
        # update attention predictor
        self.policy_att_optim.zero_grad()
        fix_policy_loss.backward()
        self.policy_att_optim.step()

        self.losses.update({'policy/total_accident': acc_policy_loss.item(),
                            'policy/actor': actor_loss.item(),
                            'policy/accident': cls_loss.item(),
                            'policy/total_fixation': fix_policy_loss.item(),
                            'policy/fixation': fix_loss.item()})
        return log_pi


    def update_entropy(self, log_pi):
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            # clip_grad_norm_(self.log_alpha, self.dim_action)  # clip gradient of log_alpha
            self.alpha_optim.step()

            # self.alpha = self.log_alpha.exp()
            self.alpha = torch.clamp_min(self.log_alpha.exp(), 0.0001)
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs

            self.losses.update({'alpha': alpha_loss.item()})
        else:
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs
        return alpha_tlogs


    def update_decoder(self, state, latent_lambda=0.0):
        # encoder
        h = self.critic.state_encoder(state)
        # decoder
        state_rec = self.decoder(h)
        # MSE reconstruction loss
        rec_loss = F.mse_loss(state, state_rec)
        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + latent_lambda * latent_loss
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()
        self.losses.update({'autoencoder': loss.item()})


    def update_parameters(self, memory, updates):
        
        # sampling from replay buffer memory
        state_batch, action_batch, reward_batch, next_state_batch, rnn_state_batch, labels_batch, mask_batch = memory.sample(self.batch_size, self.device)

        if not self.pure_sl:
            # update critic networks
            self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch, rnn_state_batch)
        
        # update actor and alpha
        alpha_values = self.alpha
        if updates % self.actor_update_interval == 0:
            log_pi = self.update_actor(state_batch, rnn_state_batch, labels_batch)

            # update entropy term
            alpha_tlogs = self.update_entropy(log_pi)
            alpha_values = alpha_tlogs.item()

        # update critic target
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # update decoder
        if self.arch_type == 'rae':
            self.update_decoder(state_batch, latent_lambda=self.latent_lambda)

        return self.losses, alpha_values


    def save_models(self, ckpt_dir, cfg, epoch):
        model_dict = {'policy_acc_model': self.policy_accident.state_dict(),
                      'policy_fix_model': self.policy_fixation.state_dict(),
                      'configs': cfg}
        torch.save(model_dict, os.path.join(ckpt_dir, 'sac_epoch_%02d.pt'%(epoch)))
        

    def load_models(self, ckpt_dir, cfg):
        if cfg.test_epoch == -1:
            filename = sorted(os.listdir(ckpt_dir))[-1]
            weight_file = os.path.join(cfg.output, 'checkpoints', filename)
        else:
            weight_file = os.path.join(cfg.output, 'checkpoints', 'sac_epoch_' + str(cfg.test_epoch) + '.pt')
        if os.path.isfile(weight_file):
            checkpoint = torch.load(weight_file, map_location=self.device)
            self.policy_accident.load_state_dict(checkpoint['policy_acc_model'])
            self.policy_fixation.load_state_dict(checkpoint['policy_fix_model'])
            print("=> loaded checkpoint '{}'".format(weight_file))
        else:
            raise FileNotFoundError
