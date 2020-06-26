import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

class ResNet_FPN(torch.nn.Module):
    def __init__(self, n_layers=50, preTrained=False):
        super(ResNet_FPN, self).__init__()
        if n_layers == 50:
            self.net = models.resnet50(pretrained=preTrained)
        else:
            raise NotImplementedError
        self.top_layer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

        self.lat_layer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.smooth_layer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth_layer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.init_new_modules()

    def init_new_modules(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.top_layer, 0, 0.01)
        normal_init(self.smooth_layer1, 0, 0.01)
        normal_init(self.smooth_layer2, 0, 0.01)
        normal_init(self.smooth_layer3, 0, 0.01)
        normal_init(self.lat_layer1, 0, 0.01)
        normal_init(self.lat_layer2, 0, 0.01)
        normal_init(self.lat_layer3, 0, 0.01)

    
    def forward(self, im):
        # block 1
        x = self.net.conv1(im)  # (B, 64, 112, 112)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        c1 = self.net.maxpool(x)  # (B, 64, 56, 56)
        # block 2, 3, 4, 5
        c2 = self.net.layer1(c1)   # (B, 256, 56, 56)
        c3 = self.net.layer2(c2)   # (B, 512, 28, 28)
        c4 = self.net.layer3(c3)   # (B, 1024, 14, 14)
        c5 = self.net.layer4(c4)   # (B, 2048, 7, 7)
        # Top down fusion
        p5 = self.top_layer(c5)    # (B, 256, 7, 7)
        # P4 block
        c4_lat = self.lat_layer1(c4)
        p4 = F.upsample(p5, size=(c4_lat.size(2), c4_lat.size(3)), mode='bilinear') + c4_lat
        p4 = self.smooth_layer1(p4)  # (B, 256, 14, 14)
        # P3 block
        c3_lat = self.lat_layer2(c3)
        p3 = F.upsample(p4, size=(c3_lat.size(2), c3_lat.size(3)), mode='bilinear') + c3_lat
        p3 = self.smooth_layer2(p3)  # (B, 256, 28, 28)
        # P2 block
        c2_lat = self.lat_layer3(c2)
        p2 = F.upsample(p3, size=(c2_lat.size(2), c2_lat.size(3)), mode='bilinear') + c2_lat
        p2 = self.smooth_layer3(p2)  # (B, 256, 56, 56)
        return p2


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



