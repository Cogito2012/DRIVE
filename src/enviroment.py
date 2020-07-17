from gym import spaces, core
import torch
import torch.nn.functional as F
from src.saliency_models import MLNet
from src.TorchFovea import TorchFovea
import numpy as np
import os


class DashCamEnv(core.Env):
    def __init__(self, shape_data, dim_action, dim_state=128, fps=30, device=torch.device("cuda")):

        self.device = device
        self.observe_model = MLNet(shape_data)
        self.output_shape = self.observe_model.output_shape
        self.foveal_model = TorchFovea(shape_data, min(shape_data)/6.0, level=5, factor=2, device=device)
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.image_size = [660, 1584]
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.fps = fps


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


    def set_data(self, video_data, focus_data, coord_data):
        """video data: (B, T, C, H, W)
            focus_data: (B, T, 1, H/8, W/8)
            coord_data: (B, T, 3), (x, y, cls)
        """ 
        assert video_data.size(0) == 1, "Only batch size == 1 is allowed!"
        # the following attributes are unchanged or ground truth of environment for an entire video
        self.video_data = torch.Tensor(video_data).to(self.device)
        self.focus_data = torch.Tensor(focus_data).to(self.device)
        self.coord_data = torch.Tensor(coord_data).to(self.device)
        self.batch_size, self.max_step, self.height, self.width = video_data.size(0), video_data.size(1), video_data.size(3), video_data.size(4)
        # fix_ctr = torch.Tensor([self.width / 2.0, self.height / 2.0]).to(torch.float32).to(device=self.device)
        # self.fixations = torch.where(self.coord_data[0, :, :2] > 0, self.coord_data[0, :, :2], fix_ctr.expand_as(self.coord_data[0, :, :2]))
        fix_ctrs = np.repeat(np.array([[self.width / 2.0, self.height / 2.0]]), self.max_step, axis=0)
        self.fixations = np.where(coord_data[0, :, :2] > 0, coord_data[0, :, :2], fix_ctrs)
        self.clsID = self.coord_data[0, :, 2].unique()[1].long() - 1  # class ID starts from 0 to 5
        # self.begin_accident = torch.clamp(torch.nonzero(self.coord_data[0, :, 2] > 0)[0, 0] / float(self.fps), min=1)
        self.begin_accident = np.maximum(np.where(coord_data[0, :, 2] > 0)[0][0] / float(self.fps), 1.0)
        # reset the agent to the initial states
        state = self.reset()
        return state


    def reset(self):
        self.cur_step = 0  # step id of the environment
        self.next_fixation = None
        # observe the first frame
        frame_data = self.video_data[:, self.cur_step]  # (B, C, H, W)
        init_fixation = torch.Tensor([self.width / 2.0, self.height / 2.0]).to(torch.int64).to(device=self.device)
        with torch.no_grad():
            self.cur_state = self.observe(frame_data, init_fixation)
        return self.cur_state.cpu().numpy()


    def observe(self, frame, fixation):
        """
        frame: (B, C, H, W)
        """ 
        # foveation
        fovea_image = self.foveal_model.foveate(frame, fixation)
        # compute saliency map
        saliency, bottom = self.observe_model(fovea_image, return_bottom=True)
        # here we use saliency map as observed states
        state = saliency * bottom  # (1, 64, 30, 40)
        max_pool = F.max_pool2d(state, kernel_size=state.size()[2:])
        avg_pool = F.avg_pool2d(state, kernel_size=state.size()[2:])
        state = torch.cat([max_pool, avg_pool], dim=1).squeeze_(dim=-1).squeeze_(dim=-1)  # (1, 128)
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


    def _exp_loss(self, pred, target, time, toa):
        '''
        :param pred:
        :param target: onehot codings for binary classification
        :param time:
        :param toa:
        :return:
        '''
        # positive example (exp_loss)
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), toa.to(pred.dtype) - time - 1)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss


    def get_reward(self, fixation_pred, accident_pred):
        """fixation_pred: (2,)
           accident_pred: (1, 1)
        """
        # attentiveness reward (mse of fixations)
        fixation_pred = self.norm_fix(fixation_pred)
        fixation_gt = self.norm_fix(self.fixations[self.cur_step + 1])
        fixation_reward = np.exp(-1.0 * np.sum(np.power(fixation_pred - fixation_gt, 2)))
        # score
        # score = F.softmax(accident_pred, dim=1)[0, 1]
        score = self.softmax(accident_pred)[1]
        if score > 0.5:
            tta_weight = (np.exp(np.maximum(self.begin_accident - self.cur_step / self.fps, 0)) - 1.0) / (np.exp(self.begin_accident) - 1.0)
            if self.clsID > 0:  
                tta_reward = 1.0 * tta_weight  # true positive
            else:
                tta_reward = 0.0  # false positive
        else:
            if self.clsID > 0:
                tta_reward = 0.0  # false negative
            else:
                tta_reward = 1.0   # true negative
                
        # reward = fixation_reward + accident_reward + tta_reward
        reward = fixation_reward + tta_reward

        return reward


    def pred_to_point(self, scale_x, scale_y):
        """Transform the predicted scaling factor ranging from -1 to 1
        into the image plane with extends=[240, 320] by considering the image padding
        """
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


    def step(self, action, with_loss=False):
        """ action: (4)
        """
        # actions input, range from -1 to 1
        self.next_fixation = self.pred_to_point(action[0], action[1])
        accident_pred = action[2:]
        
        if with_loss:
            target = torch.where(self.clsID > 0, torch.Tensor([[0, 1]]).to(self.device), torch.Tensor([[1, 0]]).to(self.device))
            cls_loss = self._exp_loss(accident_pred, target, self.cur_step / self.fps, self.begin_accident)

        if self.cur_step < self.max_step - 1:
            # reward (immediate)
            cur_reward = self.get_reward(self.next_fixation, accident_pred)
            with torch.no_grad():
                # next state
                frame_next = self.video_data[:, self.cur_step + 1]  # (B, C, H, W)
                next_state = self.observe(frame_next, torch.from_numpy(self.next_fixation).to(self.device))
            next_state = next_state.cpu().numpy()
            done = False
            info = {}
        else:
            cur_reward = 0.0
            next_state = self.cur_state.copy()
            done = True
            info = {}

        self.cur_step += 1
        self.cur_state = next_state.copy()

        if with_loss:
            return next_state, cur_reward, cls_loss, done, info
        return next_state, cur_reward, done, info

