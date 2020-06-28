import torch
from saliency_models import ResNet_FPN


class FovealVideoEnv:
    def __init__(self, device=torch.device("cuda")):
        self.device = device
        self.observation_model = ResNet_FPN(n_layers=50, preTrained=True)


    def set_data(self, video_data):
        video_data = torch.Tensor(video_data).to(self.device)
        self.batch_size, self.max_step, height, width = video_data.size(0), video_data.size(1), video_data.size(2), video_data.size(3)
        # set the initial fixation point at the center of image
        self.fixation = torch.Tensor([width / 2.0, height / 2.0]).to(torch.int64)
        # self.reset()

    def reset(self):
        self.step_id = 0  # step id of the environment
        # self.fixations = torch.zeros((self.batch_size, 2), dtype=torch.long, device=self.device)  # (B, 2)
        # self.status = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)  # (B,) done or not
        # self.is_active = torch.ones(self.batch_size, dtype=torch.uint8,  device=self.device)


    def observe(self, frame):
        featmaps = self.observation_model(frame.permute(0, 3, 1, 2))

        states = None
        return states

    def get_reward(self):
        pass

    def step(self):
        obs = self.observe()
        action = []
        return obs, action



