import torch


class FovealVideoEnv:
    def __init__(self, device=torch.device("cuda")):
        self.step_id = 0
        self.dim_observation = 376
        self.dim_action = 17
        pass

    def observe(self):
        states = None
        return states

    def get_reward(self):
        pass

    def step(self):
        obs = self.observe()
        action = []
        return obs, action


    def reset(self):
        self.step_id = 0
        pass


