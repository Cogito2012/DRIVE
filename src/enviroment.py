from gym import spaces, core
import torch
import torch.nn.functional as F
from src.saliency_models import MLNet
from src.TorchFovea import TorchFovea
import numpy as np
import os


class DashCamEnv(core.Env):
    def __init__(self, cfg, device=torch.device("cuda")):
        self.device = device
        self.observe_model = MLNet(cfg.input_shape)
        self.output_shape = self.observe_model.output_shape
        self.use_foveation = cfg.use_foveation
        if self.use_foveation:
            self.foveal_model = TorchFovea(cfg.input_shape, min(cfg.input_shape)/6.0, level=5, factor=2, device=device)
        self.image_size = cfg.image_shape
        self.dim_state = cfg.dim_state
        self.dim_action = cfg.dim_action
        self.fps = 30 / cfg.frame_interval
        self.score_thresh = cfg.score_thresh
        self.state_norm = cfg.state_norm


    def set_model(self, pretrained=False, weight_file=None):
        if pretrained and weight_file is not None:
            # load model weight file
            assert os.path.exists(weight_file), "Checkpoint directory does not exist! %s"%(weight_file)
            ckpt = torch.load(weight_file)
            self.observe_model.load_state_dict(ckpt['model'])
            self.observe_model.to(self.device)
            self.observe_model.eval()
        else:
            self.observe_model.to(self.device)
            self.observe_model.train()


    def set_data(self, video_data, coord_data):
        """ video data: (B, T, C, H, W)
            coord_data: (B, T, 3), (x, y, cls)
        """ 
        assert video_data.size(0) == 1, "Only batch size == 1 is allowed!"
        self.height, self.width = video_data.size(3), video_data.size(4)
        # the following attributes are unchanged or ground truth of environment for an entire video
        self.video_data = video_data[0].numpy()  # (T, 3, H, W)
        self.coord_data = coord_data[0].numpy()  # (T, 3)
        
        accident_inds = np.where(self.coord_data[:, 2] > 0)[0]
        if len(accident_inds) > 0:
            self.max_step = accident_inds[-1] + 1
        else:
            self.max_step = video_data.size(1)
        
        cls_set = np.unique(self.coord_data[:, 2].astype(np.int32))
        if len(cls_set) > 1:
            self.clsID = cls_set[1] # 1 or 2
            self.begin_accident = np.maximum(np.where(self.coord_data[:, 2] > 0)[0][0] / float(self.fps), 1.0)
        else:
            self.clsID = 0
            self.begin_accident = -1
        # reset the agent to the initial states
        state = self.reset()
        return state


    def reset(self):
        self.cur_step = 0  # step id of the environment
        self.next_fixation = None

        # observe the first frame
        frame_data = torch.Tensor(self.video_data[self.cur_step]).unsqueeze(0).to(self.device)  # (B, C, H, W)
        if self.use_foveation:
            # set the center coordinate as initial fixation
            init_fixation = torch.Tensor([self.width / 2.0, self.height / 2.0]).to(torch.int64).to(device=self.device)
            # foveation
            frame_data = self.foveal_model.foveate(frame_data, init_fixation)

        # observation by computing saliency
        with torch.no_grad():
            state = self.observe(frame_data)  # (1, 128)
            self.cur_state = state.cpu().numpy()
        return self.cur_state


    def observe(self, frame):
        """ 
        frame: (B, C, H, W)
        """ 
        # compute saliency map
        saliency, bottom = self.observe_model(frame, return_bottom=True)
        # here we use saliency map as observed states
        state = saliency * bottom  # (1, 64, 30, 40)
        max_pool = F.max_pool2d(state, kernel_size=state.size()[2:])
        avg_pool = F.avg_pool2d(state, kernel_size=state.size()[2:])
        state = torch.cat([max_pool, avg_pool], dim=1).squeeze_(dim=-1).squeeze_(dim=-1)  # (1, 128)
        if self.state_norm:
            state = F.normalize(state, p=2, dim=1)
        return state


    def norm_fix(self, fixation):
        fix_norm = fixation.copy()
        fix_norm[0] /= self.width
        fix_norm[1] /= self.height
        return fix_norm

    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def scales_to_point(self, scales):
        """Transform the predicted scaling factor ranging from -1 to 1
        into the image plane with extends=[240, 320] by considering the image padding
        """
        scale_x, scale_y = scales[0], scales[1]
        rows_rate = self.image_size[0] / self.height  # 660 / 240
        cols_rate = self.image_size[1] / self.width   # 1584 / 320
        if rows_rate > cols_rate:
            new_cols = (self.image_size[1] * self.height) // self.image_size[0]
            c = np.minimum(new_cols / 2.0 * (1 + scale_x), new_cols-1)  # set upper bound
            r = np.minimum(self.height / 2.0 * (1 - scale_y), self.height-1)
            c = c + (self.width - new_cols) // 2
        else:
            new_rows = (self.image_size[0] * self.width) // self.image_size[1]
            r = np.minimum(new_rows / 2.0 * (1 - scale_y), new_rows-1)
            c = np.minimum(self.width / 2.0 * (1 + scale_x), self.width-1)
            r = r + (self.height - new_rows) // 2
        point = np.array([c, r])  # (x, y)
        return point


    def point_to_scales(self, point):
        """Transform the point that is defined on [480, 640] plane into scales ranging from -1 to 1 on image plane
        point: [x, y]
        """
        rows_rate = self.image_size[0] / self.height  # 660 / 240
        cols_rate = self.image_size[1] / self.width   # 1584 / 320
        if rows_rate > cols_rate:
            new_cols = (self.image_size[1] * self.height) // self.image_size[0]
            point[0] = point[0] - (self.width - new_cols) // 2
            scale_x = (point[0] - 0.5 * new_cols) / (0.5 * new_cols)
            scale_y = (0.5 * self.height - point[1]) / (0.5 * self.height)
        else:
            new_rows = (self.image_size[0] * self.width) // self.image_size[1]
            point[1] = point[1] - (self.height - new_rows) // 2
            scale_y = (0.5 * new_rows - point[1]) / (0.5 * new_rows)
            scale_x = (point[0] - 0.5 * self.width) / (0.5 * self.width)
        scales = np.array([scale_x, scale_y])
        return scales


    def get_next_state(self, next_fixation, next_step):
        
        frame_next = torch.Tensor(self.video_data[next_step]).unsqueeze(0).to(self.device)  # (B, C, H, W)
        if self.use_foveation:
            # foveation
            next_fixation = torch.Tensor(next_fixation).to(self.device)
            frame_next = self.foveal_model.foveate(frame_next, next_fixation)

        with torch.no_grad():
            next_state = self.observe(frame_next)
        return next_state.cpu().numpy()


    def get_reward(self, fixation_pred, score):
        """ fixation_pred: (2,): image coordinates
            score: (1,): accident score
        """
        # attentiveness reward (mse of fixations)
        fixation_gt = self.coord_data[self.cur_step + 1, :2]  # (2,)
        if fixation_gt[0] > 0 and fixation_gt[1] > 0:
            # R = exp{-|P' - P|}
            # To ensure a bounded reward, the fixation points are normalized to [0, 1] with image size
            fixation_reward = np.exp(-1.0 * np.sum(np.power(self.norm_fix(fixation_pred) - self.norm_fix(fixation_gt), 2)))
        else:
            # For non-accident frames, we give zero reward
            fixation_reward = 0

        # compute the accident score
        if score > self.score_thresh:
            tta_weight = (np.exp(np.maximum(self.begin_accident - self.cur_step / self.fps, 0)) - 1.0) / (np.exp(self.begin_accident) - 1.0)
            if self.clsID > 1:
                tta_reward = 1.0 * tta_weight  # true positive
            else:
                tta_reward = 0.0  # false positive
        else:
            if self.clsID > 1:
                tta_reward = 0.0  # false negative
            else:
                tta_reward = 1.0   # true negative
        reward = fixation_reward + tta_reward
        return reward


    def step(self, action, isTraining=True):
        """ action: (3)
        """
        # actions input, range from -1 to 1
        self.next_fixation = self.scales_to_point(action[:2])

        # self.score_pred = self.softmax(action[2])[1]
        self.score_pred = 0.5 * (action[2] + 1.0)  # map to [0, 1]

        info = {'next_fixation': self.next_fixation,
                'score_pred': self.score_pred}
        
        if self.cur_step < self.max_step - 1:
            # next state
            next_state = self.get_next_state(self.next_fixation, self.cur_step + 1)
            # reward (immediate)
            cur_reward = self.get_reward(self.next_fixation, self.score_pred) if isTraining else 0
            done = False
            info.update({'gt_fixation': self.point_to_scales(self.coord_data[self.cur_step + 1, :2])})  # ground truth of the next fixation
        else:
            # The last step
            next_state = self.cur_state.copy()
            cur_reward = 0.0
            done = True
            info.update({'gt_fixation': self.point_to_scales(self.coord_data[self.cur_step, :2])})  # ground truth of the next fixation

        self.cur_step += 1
        self.cur_state = next_state.copy()

        return next_state, cur_reward, done, info

