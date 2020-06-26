import torch


class FovealVideoEnv:
    def __init__(self, device=torch.device("cuda")):
        self.device = device


    def set_data(self, video_data, focus_data, coord_data):
        video_data = torch.Tensor(video_data).to(self.device)
        focus_data = torch.Tensor(focus_data).to(self.device)
        coord_data = torch.Tensor(coord_data).to(self.device)
        self.batch_size = video_data.size(0)
        self.max_step = video_data.size(1)
        state = self.reset()
        return state

    def reset(self):
        self.step_id = 0  # step id of the environment
        self.fixations = torch.zeros((self.batch_size, 2), dtype=torch.long, device=self.device)  # (B, 2)
        # self.status = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)  # (B,) done or not
        # self.is_active = torch.ones(self.batch_size, dtype=torch.uint8,  device=self.device)
        return self.fixations


    def observe(self):
        states = None
        return states

    def get_reward(self):
        pass

    def step(self):
        obs = self.observe()
        action = []
        return obs, action



