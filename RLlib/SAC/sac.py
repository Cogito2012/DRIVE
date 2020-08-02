import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .utils import soft_update, hard_update
from .agents import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, dim_action, cfg, device=torch.device("cuda")):

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.policy_type = cfg.policy
        self.target_update_interval = cfg.target_update_interval
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning
        self.num_classes = cfg.num_classes
        self.device = device

        self.critic = QNetwork(num_inputs, dim_action, cfg.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.lr)

        self.critic_target = QNetwork(num_inputs, dim_action, cfg.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor((dim_action)).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr)

            self.policy = GaussianPolicy(num_inputs, dim_action, cfg.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=cfg.lr, weight_decay=cfg.policy_weight_decay)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, dim_action, cfg.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=cfg.lr, weight_decay=cfg.policy_weight_decay)
        
        # self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.nll_loss = torch.nn.NLLLoss(reduction='none')

    
    def set_status(self, phase='train'):
        if phase == 'train':
            self.policy.train()
            self.critic.train()
            self.critic_target.train()
        elif phase == 'eval':
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            raise NotImplementedError


    def select_action(self, state, rnn_state=None, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        if rnn_state is not None:
            rnn_state = (torch.from_numpy(rnn_state[0]).to(self.device), torch.from_numpy(rnn_state[1]).to(self.device))

        if evaluate is False:
            action, rnn_state, _, _ = self.policy.sample(state, rnn_state)
        else:
            _, rnn_state, _, action = self.policy.sample(state, rnn_state)
        
        if rnn_state is not None:
            rnn_state = torch.cat((rnn_state[0].unsqueeze(0), rnn_state[1].unsqueeze(0)), dim=0).detach().cpu().numpy()
        return action.detach().cpu().numpy()[0], rnn_state


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, rnn_state_batch, labels_batch, mask_batch = memory.sample(batch_size=batch_size)
        if rnn_state_batch[:, 0] is not None:
            rnn_state_batch = (torch.from_numpy(rnn_state_batch[:, 0]).to(self.device), torch.from_numpy(rnn_state_batch[:, 1]).to(self.device))

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        labels_batch = torch.FloatTensor(labels_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, _, next_state_log_pi, _ = self.policy.sample(next_state_batch, rnn_state_batch)
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

        pi, _, log_pi, _ = self.policy.sample(state_batch, rnn_state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # # compute the early anticipation loss
        # # accident_pred = 0.5 * torch.log((1 + pi[:, 2:]) / (1 - pi[:, 2:]) + 1e-6)
        # accident_score = torch.sqrt((1 + pi[:, 2:]) / (1 - pi[:, 2:]))
        # score_pred = accident_score / torch.sum(accident_score, dim=1, keepdim=True)
        score_pred = 0.5 * (pi[:, 2] + 1.0)

        steps_batch, clsID_batch, toa_batch, fps_batch = labels_batch[:, 0], labels_batch[:, 1], labels_batch[:, 2], labels_batch[:, 3]
        task_target = torch.zeros(batch_size, self.num_classes).to(self.device)
        task_target.scatter_(1, clsID_batch.unsqueeze(1).long(), 1)  # one-hot
        task_loss = self._exp_loss(score_pred, task_target, steps_batch / fps_batch, toa_batch)
        policy_loss = self.beta * policy_loss + task_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

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


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), task_loss.item()


    def save_models(self, ckpt_dir, cfg, epoch):
        model_dict = {'policy_model': self.policy.state_dict(),
                    'policy_optim': self.policy_optim.state_dict(),
                    'critic_model': self.critic.state_dict(),
                    'critic_target': self.critic_target.state_dict(),
                    'critic_optim': self.critic_optim.state_dict(),
                    'configs': cfg}
        if cfg.SAC.automatic_entropy_tuning:
            model_dict.update({'alpha_optim': self.critic_target.state_dict()})
        torch.save(model_dict, os.path.join(ckpt_dir, 'sac_epoch_%02d.pt'%(epoch)))
        

    def load_models(self, ckpt_dir, cfg):
        filename = sorted(os.listdir(ckpt_dir))[-1]
        weight_file = os.path.join(cfg.output, 'checkpoints', filename)
        if os.path.isfile(weight_file):
            checkpoint = torch.load(weight_file)
            self.policy.load_state_dict(checkpoint['policy_model'])
            # self.critic.load_state_dict(checkpoint['critic_model'])
            # self.critic_target.load(checkpoint['critic_target'])
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