from gym import spaces, core
import torch
import torch.nn.functional as F
from src.saliency_models import MLNet
from src.TorchFovea import TorchFovea


class DashCamEnv(core.Env):
    def __init__(self, shape_data, device=torch.device("cuda")):

        self.device = device
        self.observe_model = MLNet(shape_data).to(device)
        self.output_shape = self.observe_model.output_shape
        self.foveal_model = TorchFovea(shape_data, min(shape_data)/6.0, level=5, factor=2, device=device)
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.image_size = [660, 1584]


    def set_data(self, video_data, focus_data, coord_data, fps=30):
        """video data: (B, T, C, H, W)
            focus_data: (B, T, 1, H/8, W/8)
            coord_data: (B, T, 3), (x, y, cls)
        """ 
        assert video_data.size(0) == 1, "Only batch size == 1 is allowed!"
        self.video_data = torch.Tensor(video_data).to(self.device)
        self.focus_data = torch.Tensor(focus_data).to(self.device)
        self.coord_data = torch.Tensor(coord_data).to(self.device)
        self.batch_size, self.max_step, self.height, self.width = video_data.size(0), video_data.size(1), video_data.size(3), video_data.size(4)
        # set the initial fixation point at the center of image
        fix_ctr = torch.Tensor([self.width / 2.0, self.height / 2.0]).to(torch.float32).to(device=self.device)
        self.fixations = torch.where(self.coord_data[0, :, :2] > 0, self.coord_data[0, :, :2], fix_ctr.expand_as(self.coord_data[0, :, :2]))
        self.clsID = self.coord_data[0, :, 2].unique()[1].long() - 1  # class ID starts from 0 to 5
        self.begin_accident = torch.nonzero(self.coord_data[0, :, 2] > 0)[0, 0] / float(fps)
        self.fps = fps
        self.reset()


    def reset(self):
        self.cur_step = 0  # step id of the environment
        self.init_fixation = torch.Tensor([self.width / 2.0, self.height / 2.0]).to(torch.int64).to(device=self.device)
        # self.fixations = torch.zeros((self.batch_size, 2), dtype=torch.long, device=self.device)  # (B, 2)
        # self.status = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)  # (B,) done or not
        # self.is_active = torch.ones(self.batch_size, dtype=torch.uint8,  device=self.device)


    def observe(self, frame):
        """
        frame: (B, C, H, W)
        """ 
        self.cur_data = frame.clone()
        # foveation
        fovea_image = self.foveal_model.foveate(frame, self.init_fixation)
        # compute saliency map
        saliency, bottom = self.observe_model(fovea_image, return_bottom=True)
        # here we use saliency map as observed states
        state = saliency * bottom  # (1, 64, 30, 40)
        max_pool = F.max_pool2d(state, kernel_size=state.size()[2:])
        avg_pool = F.avg_pool2d(state, kernel_size=state.size()[2:])
        state = torch.cat([max_pool, avg_pool], dim=1).squeeze_(dim=-1).squeeze_(dim=-1)  # (1, 128)
        return state


    def norm_fix(self, fixation):
        fix_norm = fixation.clone()
        fix_norm[0] /= self.width
        fix_norm[1] /= self.height
        return fix_norm

    def get_reward(self, fixation_pred, accident_pred):
        """fixation_pred: (2,)
           accident_pred: (1, 6)
        """
        # correctness reward (classification)
        accident_reward = -1.0 * self.cls_loss(accident_pred, self.clsID.unsqueeze(0))
        # attentiveness reward (mse of fixations)
        fixation_reward = -1.0 * (self.norm_fix(fixation_pred) - self.norm_fix(self.next_fixation)).pow(2).sum().sqrt()
        # score
        score = torch.max(F.softmax(accident_pred, dim=1), dim=1)[0]
        if score > 0.5:
            tta_reward = 1.0 / (self.begin_accident.exp() - 1.0) * (torch.clamp(self.begin_accident - self.cur_step / self.fps, min=0).exp() - 1.0)
        else:
            tta_reward = 0
        reward = fixation_reward + accident_reward + tta_reward
        return reward


    def pred_to_point(self, scale_x, scale_y):
        """Transform the predicted scaling factor ranging from -1 to 1
        into the image plane with extends=[240, 320] by considering the image padding
        """
        rows_rate = self.image_size[0] / self.height  # 660 / 240
        cols_rate = self.image_size[1] / self.width   # 1584 / 320
        if rows_rate > cols_rate:
            new_cols = (self.image_size[1] * self.height) // self.image_size[0]
            c = torch.clamp(new_cols / 2.0 * (1 + scale_x), max=new_cols-1)
            r = torch.clamp(self.height / 2.0 * (1 - scale_y), max=self.height-1)
            c = c + (self.width - new_cols) // 2
        else:
            new_rows = (self.image_size[0] * self.width) // self.image_size[1]
            r = torch.clamp(new_rows / 2.0 * (1 - scale_y), max=new_rows-1)
            c = torch.clamp(self.width / 2.0 * (1 + scale_x), max=self.width-1)
            r = r + (self.height - new_rows) // 2
        point = torch.cat([c, r])  # (x, y)
        return point


    def step(self, action):
        """ action: (1, 8)
        """
        # actions input, range from -1 to 1
        fixation_pred = self.pred_to_point(action[:, 0], action[:, 1])
        # fixation_pred = torch.cat([self.width / 2.0 * (1 + action[:, 0]),
        #                            self.height / 2.0 * (1 - action[:, 1])])   # (x1, y1)
        accident_pred = action[:, 2:]
        # reward (immediate)
        reward = self.get_reward(fixation_pred, accident_pred)

        self.state = self.get_next_state(self.cur_data, fixation_pred)

        if self.step_id < self.max_step:
            done = False
            info = {}
        else:
            done = True
            info = {}

        self.step_id += 1

        return state, reward, done, info



