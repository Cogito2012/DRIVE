from gym import spaces, core
import torch
import torch.nn.functional as F
from src.saliency.mlnet import MLNet
from src.saliency.tasednet import TASED_v2
from src.soft_argmax import SpatialSoftArgmax2d
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
        self.output_shape = self.observe_model.output_shape
        self.len_clip = cfg.len_clip
        self.input_size = cfg.input_shape
        self.image_size = cfg.image_shape
        self.mask_size = cfg.mask_shape
        self.fps = 30 / cfg.frame_interval
        self.step_size = cfg.step_size
        self.score_thresh = cfg.score_thresh
        self.state_norm = cfg.state_norm
        self.rho = cfg.rho
        self.soft_argmax = SpatialSoftArgmax2d(normalized_coordinates=False)


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

        # pad the fixations defined in (480,640) into mask grid (5, 12)
        points_pad = self.down_padding_fixations(coord_data)  # (B, T, 2)
        # self.mask_data = self.point_to_mask(points_pad)
        # self.mask_data = self.mask_data.to(self.device, non_blocking=True)  # (B, T, 5, 12)
        self.points = points_pad.to(self.device, non_blocking=True)  # (B, T, 2): [r, c]

        # create weights for mask
        self.mask_weights = self.gaussian_weighing(points_pad, [0.4, 0.4])
        self.mask_weights = self.mask_weights.to(self.device, non_blocking=True)  # (B, T, 5, 12)

        # maximum number of steps
        self.max_steps = (self.video_data.size(1) - self.len_clip + 1) // self.step_size
        # neg/pos labels
        self.clsID = data_info[:, 4].to(self.device)
        self.begin_accident = data_info[:, 5].to(self.device) / float(self.fps)  # time-of-accident (seconds), for neg: toa=-1
        # reset the agent to the initial states
        state = self.reset()
        return state


    def down_padding_fixations(self, coord_data):
        """coord_data: CPU tensor (B, T, 2), defined in self.input_shape (480, 640)
        """
        # get size and ratios
        rows_rate = self.mask_size[0] / self.input_size[0]  # h ratio: 5 / 480 = 1/96
        cols_rate = self.mask_size[1] / self.input_size[1]  # w ratio: 12 / 640 = 1/53.3
        if rows_rate > cols_rate:
            new_cols = (self.mask_size[1] * self.input_size[0]) // self.mask_size[0]
            x_offset = (self.input_size[1] - new_cols) // 2
            x = torch.floor((coord_data[:, :, 0] - x_offset) * rows_rate).long()
            y = torch.floor(coord_data[:, :, 1] * rows_rate).long()
        else:
            new_rows = (self.mask_size[0] * self.input_size[1]) // self.mask_size[1]
            y_offset = (self.input_size[0] - new_rows) // 2
            x = torch.floor(coord_data[:, :, 0] * cols_rate).long()
            y = torch.floor((coord_data[:, :, 1] - y_offset) * cols_rate).long()
        pts_padded = torch.cat([y.unsqueeze(2), x.unsqueeze(2)], dim=2)  # (B, T, 2)
        pts_padded = torch.where(pts_padded > 0, pts_padded, torch.zeros_like(pts_padded))
        return pts_padded


    def point_to_mask(self, points_pad):
        """ Transform (B, T, 2) coordinates into mask maps (B, T, 5, 12)
        points_pad: [r, c]
        """
        # create mask
        batch_size, num_frames = points_pad.size(0), points_pad.size(1)
        # create indices
        batch_inds = torch.arange(batch_size).repeat_interleave(num_frames).unsqueeze(1).long()
        frame_inds = torch.arange(num_frames).repeat([batch_size]).unsqueeze(1).long()
        points_expand = torch.cat([batch_inds, frame_inds, points_pad.view(-1, 2)], dim=1)  # (B*T, 4)
        # create onehot values
        hot_vals = torch.ones((batch_size * num_frames), dtype=torch.int32)
        # generate 2D onehot masks
        masks_onehot = torch.sparse.IntTensor(points_expand.t(), hot_vals, torch.Size([batch_size, num_frames] + self.mask_size)).to_dense().float()  # (B, T, 5, 12)
        return masks_onehot


    def gaussian_weighing(self, points, sigmas):
        """Generate gaussian weights 
        point: (B, T, 2), [r, c]
        sigmas: [std_x, std_y]
        """
        batch_size, num_frames = points.size(0), points.size(1)
        weight_maps = torch.zeros((batch_size, num_frames, self.mask_size[0], self.mask_size[1]), dtype=torch.float32)
        for b in range(batch_size):
            for t in range(num_frames):
                if points[b, t, 0] > 0 and points[b, t, 1] > 0:
                    row_range = torch.arange(-points[b, t, 0], self.mask_size[0] - points[b, t, 0])
                    col_range = torch.arange(-points[b, t, 1], self.mask_size[1] - points[b, t, 1])
                    [er, ec] = torch.meshgrid(row_range, col_range)
                    dist = er**2 / (2*(sigmas[0]**2)) + ec**2 / (2.0*(sigmas[1]**2))
                    weights = torch.exp(-dist)
                    weights /= torch.sum(weights)
                    weight_maps[b, t] = weights
        return weight_maps


    def reset(self):
        self.cur_step = 0  # step id of the environment
        self.next_fixation = None
        # fetch video clip
        frame_data = self.video_data[:, self.cur_step*self.step_size: self.cur_step*self.step_size+self.len_clip]  # (B, T, C, H, W)
        # initial mask
        mask_shape = [frame_data.size(0), 1, self.mask_size[0], self.mask_size[1]]  # (B, 1, 5, 12)
        init_mask = torch.zeros(mask_shape, dtype=torch.float32).to(self.device)
        # observation by computing saliency
        with torch.no_grad():
            self.cur_state = self.observe(frame_data, init_mask)  # GPU Array(B, 124)
        return self.cur_state.detach()


    def observe(self, data_input, mask):
        """ 
        data_input: GPU tensor, (B, T, C, H, W)
        mask: GPU tensor, (B, 5, 12)
        """ 
        # up padding mask grid from 5x12 to 60x80
        saliency_td = self.up_padding(mask)
        B, T, C, H, W = data_input.size()
        if self.saliency == 'MLNet':
            # compute saliency map and volume
            data_input = data_input.view(B*T, C, H, W)
            saliency_bu, bottom = self.observe_model(data_input, return_bottom=True)
            saliency_bu = torch.sigmoid(saliency_bu)
            assert saliency_bu.size(1) == 1, "invalid saliency!"
            # saliency fusion
            saliency = self.rho * saliency_bu + (1 - self.rho) * saliency_td
            # construct states
            foveal_state = F.max_pool2d(saliency * bottom, kernel_size=bottom.size()[2:])
            if self.state_norm:
                foveal_state = F.normalize(foveal_state, p=2, dim=1)
            # down padding saliency map from 60x80 to 5x12
            salmap_state = self.down_padding(saliency)
            state = torch.cat([foveal_state.squeeze_(dim=-1).squeeze_(dim=-1), 
                               salmap_state.view(B, -1)], dim=1)  # (B, 124)
        elif self.saliency == 'TASED-Net':
            data_input = data_input.permute(0, 2, 1, 3, 4).contiguous()  # (B, C=3, T=8, H=480, W=640)
            # compute saliency map
            bottom = self.observe_model(data_input, return_bottom=True)  # (B, 192, 1, 60, 80)
            feat_foveal = F.max_pool3d(mask_padded.unsqueeze(1) * bottom, kernel_size=bottom.size()[2:])
            feat_contex = F.max_pool3d(mask_padded_inv.unsqueeze(1) * bottom, kernel_size=bottom.size()[2:])
            state = torch.cat([feat_foveal, feat_contex], dim=1).squeeze_(dim=-1).squeeze_(dim=-1).squeeze_(dim=-1)  # (B, 384)
        else:
            raise NotImplementedError
        return state

    def up_padding(self, mask):
        """mask: GPU tensor, (B, 1, 5, 12), padded to (B, 1, 60, 80)
        """
        out_shape = [mask.size(0), mask.size(1), self.output_shape[0], self.output_shape[1]]
        mask_padded = torch.zeros(out_shape, dtype=torch.float32).to(self.device)
        # get size and ratios
        rows_rate = self.mask_size[0] / self.output_shape[0]  # h ratio
        cols_rate = self.mask_size[1] / self.output_shape[1]  # w ratio
        # padding
        if rows_rate > cols_rate:
            new_cols = (self.mask_size[1] * self.output_shape[0]) // self.mask_size[0]
            mask_ctr = F.interpolate(mask, size=(self.output_shape[0], new_cols))
            mask_padded[:, :, :, ((self.output_shape[1] - new_cols) // 2):((self.output_shape[1] - new_cols) // 2 + new_cols)] = mask_ctr
        else:
            new_rows = (self.mask_size[0] * self.output_shape[1]) // self.mask_size[1]
            mask_ctr = F.interpolate(mask, size=(new_rows, self.output_shape[1]))
            mask_padded[:, :, ((self.output_shape[0] - new_rows) // 2):((self.output_shape[0] - new_rows) // 2 + new_rows), :] = mask_ctr
        return mask_padded

    def down_padding(self, saliency):
        """saliency: GPU tensor, (B, 1, 60, 80), padded to (B, 1, 5, 12)
        First, we crop the saliency so that edge blank is removed, then, resize
        """
        # get size and ratios
        rows_rate = self.mask_size[0] / self.output_shape[0]  # h ratio
        cols_rate = self.mask_size[1] / self.output_shape[1]  # w ratio
        # crop by padding
        if rows_rate > cols_rate:
            new_cols = (self.mask_size[1] * self.output_shape[0]) // self.mask_size[0]
            salcrop = saliency[:, :, :, ((self.output_shape[1] - new_cols) // 2):((self.output_shape[1] - new_cols) // 2 + new_cols)]
        else:
            new_rows = (self.mask_size[0] * self.output_shape[1]) // self.mask_size[1]
            salcrop = saliency[:, :, ((self.output_shape[0] - new_rows) // 2):((self.output_shape[0] - new_rows) // 2 + new_rows), :]
        # down sampling
        sal_down = F.interpolate(salcrop, size=self.mask_size)
        return sal_down


    def get_next_state(self, mask_pred, next_step):
        """mask_pred: (B, 1, 5, 12)
        """
        frame_data = self.video_data[:, next_step*self.step_size: next_step*self.step_size+self.len_clip]  # (B, T, C, H, W)
        with torch.no_grad():
            next_state = self.observe(frame_data, mask_pred)
        return next_state.detach()


    def get_reward(self, score_pred, mask_pred):
        """ score_pred: (B,): accident score
            mask_pred: (B, 1, 5, 12): attention mask    
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
        fix_pred = self.soft_argmax(mask_pred)  # (B, 1, 2): [x, y] (not normalized)
        fix_pred = torch.cat((fix_pred[:, :, 1], fix_pred[:, :, 0]), dim=1)  # (B, 2): [r, c]
        fix_gt = self.points[:, (self.cur_step + 1)*self.step_size, :]  # (B, 2), [r, c]
        # compute distance
        dist_sq = torch.sum(torch.pow(fix_pred.float() - fix_gt.float(), 2), dim=1, keepdim=True)
        mask_reward = fix_gt[:, 0].bool().float() * fix_gt[:, 1].bool().float()  # (B,)
        dmax = np.sqrt(np.sum(np.square(self.mask_size)))
        r_atten = mask_reward.unsqueeze(1) * torch.exp(-2.0 * dist_sq / dmax)  # If fixation exists (r>0 & c>0), reward = exp(-mse)

        # total reward
        reward_batch = r_tta + r_atten
        return reward_batch


    def step(self, actions, isTraining=True):
        """ actions: (B, 61)
        """
        batch_size = actions.size(0)
        # parse actions (current accident scores, the next attention mask)
        score_pred = 0.5 * (actions[:, 0] + 1.0)  # map to [0, 1], shape=(B, 1)
        mask_pred = actions[:, 1:].view(-1, 1, self.mask_size[0], self.mask_size[1])  # shape=(B, 1, 5, 12)
        mask_pred = 0.5 * (mask_pred + 1.0)

        if self.cur_step < self.max_steps - 1:  # cur_step starts from 0
            # next state
            next_state = self.get_next_state(mask_pred, self.cur_step + 1)
            # reward (immediate)
            cur_rewards = self.get_reward(score_pred, mask_pred) if isTraining else 0
        else:
            # The last step
            next_state = self.cur_state.clone()  # GPU array
            cur_rewards = torch.zeros([batch_size, 1], dtype=torch.float32).to(self.device) if isTraining else 0

        self.cur_step += 1
        self.cur_state = next_state.clone()

        return next_state, cur_rewards

