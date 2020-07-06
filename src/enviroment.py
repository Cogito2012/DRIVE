from gym import spaces, core
import torch
import torch.nn.functional as F
from src.saliency_models import MLNet
from src.TorchFovea import TorchFovea


class DashCamEnv(core.Env):
    def __init__(self, shape_data, device=torch.device("cuda")):

        self.device = device
        # self.observe_model = ResNet_FPN(n_layers=50, preTrained=True)
        self.observe_model = MLNet(shape_data).to(device)
        self.output_shape = self.observe_model.output_shape
        self.foveal_model = TorchFovea(shape_data, min(shape_data)/6.0, level=5, factor=2, device=device)


    def set_data(self, video_data):
        """video data: (B, T, C, H, W)
        """ 
        video_data = torch.Tensor(video_data).to(self.device)
        self.batch_size, self.max_step, height, width = video_data.size(0), video_data.size(1), video_data.size(3), video_data.size(4)
        # set the initial fixation point at the center of image
        self.fixation = torch.Tensor([width / 2.0, height / 2.0]).to(torch.int64).to(device=self.device)
        self.reset()

    def reset(self):
        self.step_id = 0  # step id of the environment
        # self.fixations = torch.zeros((self.batch_size, 2), dtype=torch.long, device=self.device)  # (B, 2)
        # self.status = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)  # (B,) done or not
        # self.is_active = torch.ones(self.batch_size, dtype=torch.uint8,  device=self.device)


    def observe(self, frame):
        """
        frame: (B, C, H, W)
        """ 
        self.cur_data = frame.clone()
        # foveation
        fovea_image = self.foveal_model.foveate(frame, self.fixation)
        # compute saliency map
        saliency, bottom = self.observe_model(fovea_image, return_bottom=True)
        # here we use saliency map as observed states
        state = saliency * bottom  # (1, 64, 30, 40)
        max_pool = F.max_pool2d(state, kernel_size=state.size()[2:])
        avg_pool = F.avg_pool2d(state, kernel_size=state.size()[2:])
        state = torch.cat([max_pool, avg_pool], dim=1).squeeze_()  # (128,)
        return state


    def get_reward(self):
        pass

    def step(self, action):
        """ action: (12,)
        """
        # actions input 
        accident = action[:2]
        fix_ang = action[2:10]
        fix_amp = action[10:]

        # reward (immediate)
        reward = self.get_reward()

        self.state = self.get_next_state(self.cur_data, action)

        if self.step_id < self.max_step:
            done = False
            info = {}
        else:
            done = True
            info = {}

        self.step_id += 1

        return state, reward, done, info



