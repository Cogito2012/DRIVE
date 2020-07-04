import torch
from src.saliency_models import MLNet
from src.TorchFovea import TorchFovea

class FovealVideoEnv:
    def __init__(self, input_shape, device=torch.device("cuda")):
        self.device = device
        # self.observe_model = ResNet_FPN(n_layers=50, preTrained=True)
        self.observe_model = MLNet(input_shape).to(device)
        self.output_shape = self.observe_model.output_shape
        self.foveal_model = TorchFovea(input_shape, min(input_shape)/6.0, level=5, factor=2, device=device)


    def set_data(self, video_data):
        """video data: (B, T, C, H, W)
        """ 
        video_data = torch.Tensor(video_data).to(self.device)
        self.batch_size, self.max_step, height, width = video_data.size(0), video_data.size(1), video_data.size(3), video_data.size(4)
        # set the initial fixation point at the center of image
        self.fixation = torch.Tensor([width / 2.0, height / 2.0]).to(torch.int64).to(device=self.device)
        # self.reset()

    def reset(self):
        self.step_id = 0  # step id of the environment
        # self.fixations = torch.zeros((self.batch_size, 2), dtype=torch.long, device=self.device)  # (B, 2)
        # self.status = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)  # (B,) done or not
        # self.is_active = torch.ones(self.batch_size, dtype=torch.uint8,  device=self.device)


    def observe(self, frame):
        """
        frame: (B, C, H, W)
        """ 
        # foveation
        fovea_image = self.foveal_model.foveate(frame, self.fixation)
        # compute saliency map
        featmaps = self.observe_model(fovea_image)

        # fovea_image = fovea_image.permute(0, 2, 3, 1)  # (B, H, W, C)
        # foveated_image = fovea_image[0].cpu().detach().numpy()
        # import cv2
        # cv2.imwrite("result1.png", foveated_image)

        states = None
        return states

    def get_reward(self):
        pass

    def step(self):
        obs = self.observe()
        action = []
        return obs, action



