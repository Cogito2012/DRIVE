import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
# from torch.nn.utils import clip_grad_norm_
from .utils import soft_update, hard_update
from .agents import GaussianPolicy, QNetwork, DeterministicPolicy, StateDecoder, AttentionPolicy
import time


class SAC(object):
    def __init__(self, mask_size, sal_size, cfg, device=torch.device("cuda")):

        self.mask_size = mask_size
        self.sal_size = sal_size
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.beta_actor = cfg.beta_actor
        self.beta_fix = cfg.beta_fix

        self.arch_type = cfg.arch_type
        self.policy_type = cfg.policy
        self.target_update_interval = cfg.target_update_interval
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning
        self.num_classes = cfg.num_classes
        self.device = device
        self.batch_size = cfg.batch_size

        # state dims
        self.dim_state_fovea = cfg.dim_state_fovea
        self.dim_state_contex = cfg.dim_state_contex
        self.dim_state_atten = cfg.dim_state_atten
        self.dim_state = cfg.dim_state_fovea + cfg.dim_state_contex + cfg.dim_state_atten
        # action dims
        self.dim_action_accident = cfg.dim_action_accident
        self.dim_action_attention = cfg.dim_action_attention
        self.dim_action = cfg.dim_action_accident + cfg.dim_action_attention

        # create actor and critics
        self.policy_accident, self.policy_attention, self.critic, self.critic_target = self.create_actor_critics(cfg)
        hard_update(self.critic_target, self.critic)

        # optimizers
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.lr)
        self.params_actor = [p for p in self.policy_accident.parameters()] + [p for p in self.policy_attention.parameters()]
        self.policy_optim = Adam(self.params_actor, lr=cfg.lr, weight_decay=cfg.weight_decay)
        # self.policy_att_optim = Adam(self.policy_attention.parameters(), lr=cfg.lr)

        if cfg.arch_type == 'rae':
            # the encoder is shared between actor and critic, weights need to be tied
            self.policy_accident.state_encoder.copy_conv_weights_from(self.critic.state_encoder)
            # decoder
            self.decoder = StateDecoder(cfg.dim_latent, self.dim_state_fovea).to(device=self.device)
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optim = Adam(self.critic.state_encoder.parameters(), lr=cfg.lr)
            # optimizer for decoder
            self.decoder_optim = Adam(self.decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.latent_lambda = cfg.latent_lambda
        
        if self.policy_type == "Gaussian" and self.automatic_entropy_tuning:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -torch.prod(torch.Tensor((self.dim_action_accident)).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
        
        # self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.nll_loss = torch.nn.NLLLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')


    def create_actor_critics(self, cfg):
        # create critic networks
        critic = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(device=self.device)
        critic_target = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(self.device)
        # create accident anticipation policy
        if self.policy_type == "Gaussian":
            policy_accident = GaussianPolicy(self.dim_state_fovea, self.dim_action_accident, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(self.device)
        else:
            policy = DeterministicPolicy(self.dim_state_fovea, self.dim_action_accident, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(self.device)
        # create attention mask prediction policy
        policy_attention = AttentionPolicy(self.dim_state_contex, self.dim_state_atten, self.dim_action_attention, cfg.hidden_size, self.mask_size, self.sal_size).to(self.device)
        return policy_accident, policy_attention, critic, critic_target

    
    def set_status(self, phase='train'):
        isTraining = True if phase == 'train' else False
        self.policy_accident.train(isTraining)
        self.policy_attention.train(isTraining)
        self.critic.train(isTraining)
        self.critic_target.train(isTraining)
        if self.arch_type == 'rae':
            self.decoder.train(isTraining) 


    def select_action(self, state, rnn_state=None, evaluate=False):
        """state: (B, 64+64+60), [fovea_state, contex_state, mask_state]
        """
        foveal_state = state[:, :self.dim_state_fovea]
        contex_state = state[:, self.dim_state_fovea: self.dim_state_fovea + self.dim_state_contex]
        mask_state = state[:, self.dim_state_fovea + self.dim_state_contex:]
        # execute actions
        if evaluate is False:
            action_acc, rnn_state, _, _ = self.policy_accident.sample(foveal_state, rnn_state)
        else:
            _, rnn_state, _, action_acc = self.policy_accident.sample(foveal_state, rnn_state)
        action_att = self.policy_attention.forward(contex_state, mask_state)
        # get actions
        actions = torch.cat([action_acc.detach(), action_att.detach()], dim=1)
        if rnn_state is not None:
            rnn_state = (rnn_state[0].detach(), rnn_state[1].detach())
        return actions, rnn_state


    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, rnn_state_batch):
        with torch.no_grad():
            # split the next_states
            next_fovea_state = next_state_batch[:, :self.dim_state_fovea]
            next_contex_state = next_state_batch[:, self.dim_state_fovea: self.dim_state_fovea + self.dim_state_contex]
            next_mask_state = next_state_batch[:, self.dim_state_fovea + self.dim_state_contex:]
            # inference two policies
            next_accid_state_action, _, next_state_log_pi, _ = self.policy_accident.sample(next_fovea_state, rnn_state_batch)
            next_atten_state_action = self.policy_attention.forward(next_contex_state, next_mask_state)
            next_state_action = torch.cat([next_accid_state_action, next_atten_state_action], dim=1)
            # interence critics
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        return qf_loss


    def update_actor(self, state_batch, rnn_state_batch):
        # split the states
        fovea_state = state_batch[:, :self.dim_state_fovea]
        contex_state = state_batch[:, self.dim_state_fovea: self.dim_state_fovea + self.dim_state_contex]
        mask_state = state_batch[:, self.dim_state_fovea + self.dim_state_contex:]

        # sampling
        pi, _, log_pi, mean_action = self.policy_accident.sample(fovea_state, rnn_state_batch, detach=True)
        atten_state_action = self.policy_attention.forward(contex_state, mask_state)
        pi = torch.cat([pi, atten_state_action], dim=1)

        qf1_pi, qf2_pi = self.critic(state_batch, pi, detach=True)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # actor loss
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # update accident predictor
        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        return log_pi, actor_loss


    def update_entropy(self, log_pi):
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs
        return alpha_loss, alpha_tlogs


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

        return loss


    def update_parameters(self, memory, updates):
        
        # sampling from replay buffer memory
        state_batch, action_batch, reward_batch, next_state_batch, rnn_state_batch, mask_batch = memory.sample(self.batch_size, self.device)

        # update critic networks
        qf_loss = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch, rnn_state_batch)
        
        # update actor and alpha
        log_pi, policy_loss = self.update_actor(state_batch, rnn_state_batch)

        # update entropy term
        alpha_loss, alpha_tlogs = self.update_entropy(log_pi)

        # update critic target
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # update decoder
        if self.arch_type == 'rae':
            loss_ae = self.update_decoder(state_batch, latent_lambda=self.latent_lambda)

        # gather results
        losses = {'critic': qf_loss.item(), 
                  'policy': policy_loss.item(),
                  'alpha': alpha_loss.item()}
        if self.arch_type == 'rae':
            losses.update({'autoencoder': loss_ae.item()})
        alpha_values = alpha_tlogs.item()
        return losses, alpha_values


    def save_models(self, ckpt_dir, cfg, epoch):
        model_dict = {'policy_acc_model': self.policy_accident.state_dict(),
                    'policy_att_model': self.policy_attention.state_dict(),
                    'configs': cfg}
        if cfg.SAC.automatic_entropy_tuning:
            model_dict.update({'alpha_optim': self.critic_target.state_dict()})
        torch.save(model_dict, os.path.join(ckpt_dir, 'sac_epoch_%02d.pt'%(epoch)))
        

    def load_models(self, ckpt_dir, cfg):
        if cfg.test_epoch == -1:
            filename = sorted(os.listdir(ckpt_dir))[-1]
            weight_file = os.path.join(cfg.output, 'checkpoints', filename)
        else:
            weight_file = os.path.join(cfg.output, 'checkpoints', 'sac_epoch_' + str(cfg.test_epoch) + '.pt')
        if os.path.isfile(weight_file):
            checkpoint = torch.load(weight_file)
            self.policy_accident.load_state_dict(checkpoint['policy_acc_model'])
            self.policy_attention.load_state_dict(checkpoint['policy_att_model'])
            print("=> loaded checkpoint '{}'".format(weight_file))
        else:
            raise FileNotFoundError


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

    
    def _fixation_loss(self, pred_fix, gt_fix):
        # Mask out the fixations of accident frames
        mask = gt_fix[:,0].gt(0.5)
        pred_fix = pred_fix[mask]
        gt_fix = gt_fix[mask]
        if pred_fix.size(0) > 0 and gt_fix.size(0) > 0:
            # MSE loss
            loss = torch.mean(self.mse_loss(pred_fix, gt_fix))
        else:
            loss = torch.tensor(0.).to(self.device)
        return loss