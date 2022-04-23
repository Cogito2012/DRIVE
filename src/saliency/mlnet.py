import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import math

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
        p4 = F.interpolate(p5, size=(c4_lat.size(2), c4_lat.size(3)), mode='bilinear') + c4_lat
        p4 = self.smooth_layer1(p4)  # (B, 256, 14, 14)
        # P3 block
        c3_lat = self.lat_layer2(c3)
        p3 = F.interpolate(p4, size=(c3_lat.size(2), c3_lat.size(3)), mode='bilinear') + c3_lat
        p3 = self.smooth_layer2(p3)  # (B, 256, 28, 28)
        # P2 block
        c2_lat = self.lat_layer3(c2)
        p2 = F.interpolate(p3, size=(c2_lat.size(2), c2_lat.size(3)), mode='bilinear') + c2_lat
        p2 = self.smooth_layer3(p2)  # (B, 256, 56, 56)
        return p2


class MLNet(nn.Module):
    """
    Referenced from: https://github.com/immortal3/MLNet-Pytorch/blob/master/MLNet_Pytorch.ipynb
    """
    def __init__(self, input_shape):
        super(MLNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = [int(input_shape[0] / 8), int(input_shape[1] / 8)]
        self.scale_factor = 10
        self.prior_size = [int(self.output_shape[0] / self.scale_factor), int(self.output_shape[1] / self.scale_factor)]

        # loading pre-trained vgg16 model and removing last max pooling layer (Conv5-3 pooling)
        # 16: conv3-3 pool (1/8), 23: conv4-3 pool (1/16), 30: conv5-3 (1/16)
        vgg16_model = models.vgg16(pretrained = True)
        self.freeze_params(vgg16_model, 21)
        features = list(vgg16_model.features)[:-1]
        
        # making same spatial size  by calculation :) 
        # in pytorch there was problem outputing same size in maxpool2d
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2

        self.features = nn.ModuleList(features).eval()
        # adding dropout layer
        self.fddropout = nn.Dropout2d(p=0.5)
        # adding convolution layer to down number of filters 1280 ==> 64
        self.int_conv = nn.Conv2d(1280, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pre_final_conv = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1) ,padding=(0, 0))
        # prior initialized to ones
        self.prior = nn.Parameter(torch.ones((1, 1, self.prior_size[0], self.prior_size[1]), requires_grad=True))
        
        # bilinear upsampling layer
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        # initialize new parameters
        self.init_new_params()
        

    def freeze_params(self, model, last_freeze_layer):
        # freezing Layer
        for i, param in enumerate(model.parameters()):
            if i <= last_freeze_layer:
                param.requires_grad = False


    def init_new_params(self):

        def zero_params(tensor):
            if tensor is not None:
                tensor.data.fill_(0)
        
        nn.init.kaiming_normal_(self.int_conv.weight, mode='fan_out', nonlinearity='relu')
        zero_params(self.int_conv.bias)
        nn.init.kaiming_normal_(self.pre_final_conv.weight, mode='fan_out', nonlinearity='relu')
        zero_params(self.pre_final_conv.bias)
        torch.nn.init.xavier_normal_(self.prior)


    def forward(self, x, return_bottom=False):
        results = []
        for ii, model in enumerate(self.features):
            # model = model.to(x.device)
            x = model(x)
            if ii in {16,23,29}:
                results.append(x)
        
        # concat to get 1280 = 512 + 512 + 256
        x = torch.cat((results[0],results[1],results[2]),1) 
        
        # adding dropout layer with dropout set to 0.5 (default)
        x = self.fddropout(x)
        
        # 64 filters convolution layer
        bottom = self.int_conv(x)
        # 1*1 convolution layer
        x = self.pre_final_conv(bottom)

        upscaled_prior = self.bilinearup(self.prior)

        # dot product with prior
        x = x * upscaled_prior
        # x = torch.sigmoid(x)
        x = torch.nn.functional.relu(x,inplace=True)

        if return_bottom:
            return x, bottom
        return x

    
# Modified MSE Loss Function
class ModMSELoss(torch.nn.Module):
    def __init__(self, shape_gt):
        super(ModMSELoss, self).__init__()
        self.shape_r_gt = shape_gt[0]
        self.shape_c_gt = shape_gt[1]
        
    def forward(self, output , label , prior):
        prior_size = prior.shape
        output_max = torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],self.shape_r_gt,self.shape_c_gt)
        reg = ( 1.0/(prior_size[0]*prior_size[1]) ) * ( 1 - prior)**2  # (1, 1, 6, 8)
        loss = torch.mean( ((output / (output_max + 1e-6) - label) / (1 - label + 0.1))**2)  +  torch.sum(reg)
        return loss