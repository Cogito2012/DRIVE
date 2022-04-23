from gym import spaces, core
import torch
import torch.nn.functional as F
from src.saliency.mlnet import MLNet
from src.saliency.tasednet import TASED_v2
from src.TorchFovea import TorchFovea
from src.data_transform import scales_to_point, norm_fix
from metrics.losses import fixation_loss
import numpy as np
import os, cv2
import time


class DashCamEnv(core.Env):
    def __init__(self, cfg, device=torch.device("cuda")):
        self.device = device
        self.saliency = cfg.saliency
        if self.saliency == 'MLNet':
            self.observe_model = MLNet(cfg.input_shape)
        elif self.saliency == 'TASED-Net':
            self.observe_model = TASED_v2(cfg.input_shape)
        else:
            raise NotImplementedError
        self.output_shape = self.observe_model.output_shape  # (60, 80)
        self.foveal_model = TorchFovea(cfg.input_shape, min(cfg.input_shape)/6.0, level=5, factor=2, device=device)
        self.len_clip = cfg.len_clip
        self.input_size = cfg.input_shape  # (480, 640)
        self.image_size = cfg.image_shape  # (330, 792)
        self.fps = 30 / cfg.frame_interval
        self.step_size = cfg.step_size
        self.score_thresh = cfg.score_thresh
        self.state_norm = cfg.state_norm
        self.fusion = cfg.fusion
        self.fusion_margin = cfg.fusion_margin
        self.rho = cfg.rho
        self.use_salmap = cfg.use_salmap


    def set_model(self, pretrained=False, weight_file=None):
        if pretrained and weight_file is not None:
            # load model weight file
            assert os.path.exists(weight_file), "Checkpoint directory does not exist! %s"%(weight_file)
            ckpt = torch.load(weight_file)
            if self.saliency == 'MLNet':
                self.observe_model.load_state_dict(ckpt['model'])
            elif self.saliency == 'TASED-Net':
                model_dict = self.observe_model.state_dict()
                for name, param in ckpt.items():
                    if 'module' in name:
                        name = '.'.join(name.split('.')[1:])
                    if name in model_dict:
                        if param.size() == model_dict[name].size():
                            model_dict[name].copy_(param)
                        else:
                            print (' size? ' + name, param.size(), model_dict[name].size())
                    else:
                        print (' name? ' + name)
                self.observe_model.load_state_dict(model_dict)
            else:
                raise NotImplementedError
            self.observe_model.to(self.device)
            self.observe_model.eval()
        else:
            self.observe_model.to(self.device)
            self.observe_model.train()


    def set_data(self, video_data, coord_data, data_info):
        """ video data: (B, T, C, H, W)
            coord_data: (B, T, 2), (x, y)
            data_info: (B, 6), (accID, vid, clip_start, clip_end, label, toa)
        """ 
        # the following attributes are unchanged or ground truth of environment for an entire video
        self.video_data = video_data.float().to(self.device, non_blocking=True)  # (B, T, 3, H, W)
        self.coord_data = coord_data.float().to(self.device, non_blocking=True)  # (B, T, 2): (x,y)

        # maximum number of steps
        self.max_steps = (self.video_data.size(1) - self.len_clip + 1) // self.step_size
        # neg/pos labels
        self.clsID = data_info[:, 4].to(self.device)
        self.begin_accident = data_info[:, 5].to(self.device) / float(self.fps)  # time-of-accident (seconds), for neg: toa=-1
        # reset the agent to the initial states
        state = self.reset()
        return state


    def reset(self):
        self.cur_step = 0  # step id of the environment
        self.next_fixation = None
        if self.use_salmap:
            self.cur_saliency = None
        # fetch video clip
        frame_data = self.video_data[:, self.cur_step*self.step_size: self.cur_step*self.step_size+self.len_clip]  # (B, T, C, H, W)
        # observation by computing saliency
        with torch.no_grad():
            self.cur_state = self.observe(frame_data, fixation=None)  # GPU Array(B, 124)
        return self.cur_state.detach()


    def minmax_norm(self, salmap):
        """Normalize the saliency map with min-max
        salmap: (B, 1, H, W)
        """
        batch_size, height, width = salmap.size(0), salmap.size(2), salmap.size(3)
        salmap_data = salmap.view(batch_size, -1)  # (B, H*W)
        min_vals = salmap_data.min(1, keepdim=True)[0]  # (B, 1)
        max_vals = salmap_data.max(1, keepdim=True)[0]  # (B, 1)
        salmap_norm = (salmap_data - min_vals) / (max_vals - min_vals + 1e-6)
        salmap_norm = salmap_norm.view(batch_size, 1, height, width)
        return salmap_norm



    def observe(self, data_input, fixation=None):
        """ 
        data_input: GPU tensor, (B, T, C, H, W)
        fixation: GPU tensor, (B, 2)
        """ 
        B, T, C, H, W = data_input.size()
        if self.saliency == 'MLNet':
            data_input = data_input.view(B*T, C, H, W)
            # compute bottom-up saliency map
            saliency_bu, bottom = self.observe_model(data_input, return_bottom=True)  # (B, 1, H, W)
            saliency_bu = self.minmax_norm(saliency_bu)
            assert saliency_bu.size(1) == 1, "invalid saliency!"
            # compute top-down saliency map
            if fixation is not None:
                data_foveal = self.foveal_model.foveate(data_input, fixation)
                saliency_td = self.observe_model(data_foveal)  # (B, 1, H, W)
                saliency_td = self.minmax_norm(saliency_td)
                # saliency fusion
                rho = self.rho if self.fusion == 'static' else self.rho.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 1)
                saliency = (1 - rho) * saliency_bu + rho * saliency_td
            else:
                saliency = saliency_bu.clone()
            # construct states
            state_max = F.max_pool2d(saliency * bottom, kernel_size=bottom.size()[2:]).squeeze_(dim=-1).squeeze_(dim=-1)
            state_avg = F.avg_pool2d(saliency * bottom, kernel_size=bottom.size()[2:]).squeeze_(dim=-1).squeeze_(dim=-1)
        elif self.saliency == 'TASED-Net':
            data_observe = data_input.permute(0, 2, 1, 3, 4).contiguous()  # (B, C=3, T=32, H=480, W=640)
            # compute saliency map
            saliency_bu, bottom = self.observe_model(data_observe, return_bottom=True)  # (1,1,1,60,80), [1, 192, 4, 60, 80]
            saliency_bu = self.minmax_norm(saliency_bu.squeeze(2)).unsqueeze(2)
            # compute top-down saliency map
            if fixation is not None:
                data_foveal = self.foveal_model.foveate(data_input[:, 0], fixation)
                saliency_td = self.observe_model(data_foveal.unsqueeze(2).repeat(1, 1, T, 1, 1))  # (B, 1, 1, 60, 80)
                saliency_td = self.minmax_norm(saliency_td.squeeze(2)).unsqueeze(2)
                # saliency fusion
                rho = self.rho if self.fusion == 'static' else self.rho.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 1, 1)
                saliency = (1 - rho) * saliency_bu + rho * saliency_td
            else:
                saliency = saliency_bu.clone()
            # construct states
            state_max = F.max_pool3d(saliency * bottom, kernel_size=bottom.size()[2:]).squeeze_(dim=-1).squeeze_(dim=-1).squeeze_(dim=-1)
            state_avg = F.avg_pool3d(saliency * bottom, kernel_size=bottom.size()[2:]).squeeze_(dim=-1).squeeze_(dim=-1).squeeze_(dim=-1)
        else:
            raise NotImplementedError
        # normalize state representation
        if self.state_norm:
            state_max = F.normalize(state_max, p=2, dim=1)
            state_avg = F.normalize(state_avg, p=2, dim=1)
        state = torch.cat([state_max, state_avg], dim=1)  # (B, 128)
        if self.use_salmap:
            self.cur_saliency = saliency.clone()
        return state


    def get_next_state(self, fixation_next, next_step):
        """mask_pred: (B, 2)
        """
        frame_data = self.video_data[:, next_step*self.step_size: next_step*self.step_size+self.len_clip]  # (B, T, C, H, W)
        with torch.no_grad():
            next_state = self.observe(frame_data, fixation_next)
        return next_state.detach()


    def get_reward(self, score_pred, fix_pred):
        """ score_pred: (B,): accident score
            fix_pred: (B, 2): fixation scales, defined in (480, 640)
        """
        # compute the time-to-accident reward
        batch_size = score_pred.size(0)
        exp_term = torch.clamp(self.begin_accident - self.cur_step*self.step_size / self.fps, min=0)  # max(0, ta-t)
        tta_weights = (torch.exp(exp_term) - 1.0) / (torch.exp(self.begin_accident) - 1.0)
        tta_weights = tta_weights.float()
        # compute XNOR distance between pred and gt
        cls_pred = (score_pred > self.score_thresh).int()
        xnor_dist = torch.logical_not(torch.logical_xor(cls_pred, self.clsID)).float()
        r_tta = (tta_weights * xnor_dist).unsqueeze(1)

        # compute fixation prediction award
        fix_gt = self.coord_data[:, (self.cur_step + 1)*self.step_size, :]  # (B, 2), [x, y]
        # compute distance
        dist_sq = torch.sum(torch.pow(norm_fix(fix_pred, self.input_size) - norm_fix(fix_gt, self.input_size), 2), dim=1, keepdim=True)  # [0, sqrt(2)]
        mask_reward = fix_gt[:, 0].bool().float() * fix_gt[:, 1].bool().float()  # (B,)
        r_atten = mask_reward.unsqueeze(1) * torch.exp(-10.0 * dist_sq)  # If fixation exists (r>0 & c>0), reward = exp(-mse)

        # total reward
        reward_batch = r_tta + r_atten
        return reward_batch


    def step(self, actions, isTraining=True):
        """ actions: (B, 3)
        """
        batch_size = actions.size(0)
        # parse actions (current accident scores, the next attention mask)
        score_pred = 0.5 * (actions[:, 0] + 1.0)  # map to [0, 1], shape=(B,)
        fix_pred = scales_to_point(actions[:, 1:], self.image_size, self.input_size)  # (B, 2)  (x,y)

        # update rho (dynamic)
        if self.fusion == 'dynamic':
            self.rho = torch.clamp_max(score_pred.clone(), self.fusion_margin)  # (B,)

        info = {}
        if not isTraining:
            info.update({'pred_score': score_pred, 'pred_fixation': fix_pred})

        if self.cur_step < self.max_steps - 1:  # cur_step starts from 0
            # next state
            next_state = self.get_next_state(fix_pred, self.cur_step + 1)
            # reward (immediate)
            cur_rewards = self.get_reward(score_pred, fix_pred) if isTraining else 0
        else:
            # The last step
            next_state = self.cur_state.clone()  # GPU array
            cur_rewards = torch.zeros([batch_size, 1], dtype=torch.float32).to(self.device) if isTraining else 0

        self.cur_step += 1
        self.cur_state = next_state.clone()

        return next_state, cur_rewards, info

