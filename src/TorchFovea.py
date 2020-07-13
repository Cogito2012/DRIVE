import torch
import math
import torch.nn.functional as F
from kornia import PyrDown, PyrUp

class TorchFovea(torch.nn.Module):
    def __init__(self, imgsize, sigma, level=5, factor=2.0, device=torch.device('cuda')):
        super(TorchFovea).__init__()
        assert len(imgsize) == 2, "Invalid image size!"
        self.imgsize = imgsize  # (height, width)
        assert sigma > 0, "Invalid sigma!"
        self.sigma = sigma
        self.level = level
        self.factor = factor
        self.device = device
        # create Pyramid Filters
        self.filters = self.create_pyramid_filters()
        self.filter_sizes = [torch.Tensor([self.filters[i].size(1), self.filters[i].size(0)]).to(device) for i in range(self.level)]


    def createFilter(self, height, width):
        s = 2.0 * self.sigma * self.sigma
        xc = int(round(width * 0.5))
        yc = int(round(height * 0.5))

        [ex, ey] = torch.meshgrid(torch.arange(-xc, width - xc - 1).to(device=self.device), 
                                  torch.arange(-yc, height - yc - 1).to(device=self.device))
        d = torch.pow(ex, 2) + torch.pow(ey, 2)
        gDist = torch.exp(-d / s)
        maxValue = torch.max(gDist)
        gKernel = torch.transpose(gDist / maxValue, 0, 1)  # (1, H, W)
        return gKernel


    def create_pyramid_filters(self):
        m = int(math.floor(self.factor * self.imgsize[0]))
        n = int(math.floor(self.factor * self.imgsize[1]))
        # create kernel pyramid
        kernels = [self.createFilter(m, n)]
        for i in range(self.level):
            kernel_down = PyrDown()(kernels[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            kernels.append(kernel_down)
        return kernels


    def create_pyramid_images(self, images):
        image_pyrmid = []
        curImg = images.clone()
        for i in range(self.level):
            im_down = PyrDown()(curImg)
            im_up = PyrUp()(im_down)
            im_up = F.interpolate(im_up, [curImg.size(2), curImg.size(3)])
            im_lap = curImg - im_up
            image_pyrmid.append(im_lap)
            curImg = im_down
        return image_pyrmid, im_up


    def foveate(self, images, fixations):
        """
        images: (B, C, H, W)
        fixations: (B, 2)
        """
        # build image pyramid
        batchsize, channel = images.size(0), images.size(1)
        image_pyrmid, im_smallest = self.create_pyramid_images(images)
        image_sizes = [[image_pyrmid[i].size(3), image_pyrmid[i].size(2)] for i in range(self.level)]

        # foveation
        fovea_image = im_smallest.clone()
        for i in range(self.level-1, -1, -1):
            fix = fixations / math.pow(2, i)
            rect = self.filter_sizes[i].repeat(images.size(0), 1) / 2.0 - fix
            rect = rect.to(torch.int64)  # (B, 2)
            # crop the filter
            x1 = rect[:, 0]
            x2 = rect[:, 0] + image_sizes[i][0] 
            y1 = rect[:, 1]
            y2 = rect[:, 1] + image_sizes[i][1]
            kernel_rect = [self.filters[i][y1[j]:y2[j], x1[j]:x2[j]] for j in range(batchsize)]
            kernel_rect = torch.stack(kernel_rect).unsqueeze(1)  # expand channel dimension  # (B, 1, 480, 639)
            # filtering
            kernel_rect = F.interpolate(kernel_rect, [image_sizes[i][1], image_sizes[i][0]])
            im_filtered = image_pyrmid[i] * kernel_rect.repeat(1, channel, 1, 1)
            if i!=self.level - 1:
                fovea_image = PyrUp()(fovea_image)
                fovea_image = F.interpolate(fovea_image, [im_filtered.size(2), im_filtered.size(3)])
            # take summation
            fovea_image += im_filtered
        return fovea_image

    
if __name__ == "__main__":
    import cv2
    import numpy as np
    import time
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    im = cv2.imread('src/sample.png')
    height, width, channel = im.shape

    sigma = width/6.0  # param of gaussian kernel
    levels = 5   # number of gaussian pyramid levels
    factor = 2.0  # larger than 1.0

    # prepare input images
    im = np.transpose(np.float32(im), [2, 0, 1])  # (C, H, W)
    images = torch.stack((torch.from_numpy(im), torch.from_numpy(im))).to(device)  # (B, C, H, W)
    # prepare input fixations
    fix = np.array([[450, 410], [850, 300]])  # (B, 2)
    fixations = torch.from_numpy(fix).to(device)  # (B, 2)

    # perform image foveation
    t_start = time.time()
    fovea_layer = TorchFovea((height, width), sigma, level=levels, factor=factor, device=device)
    fovea_image = fovea_layer.foveate(images, fixations)
    print("Ellapsed time: %.6f"%(time.time()-t_start))
    fovea_image = fovea_image.permute(0, 2, 3, 1)  # (B, H, W, C)
    
    # # write image 2
    # foveated_image = fovea_image[0].cpu().detach().numpy()
    # cv2.imwrite("result1.png", foveated_image)
    # # write image 1
    # foveated_image = fovea_image[1].cpu().detach().numpy()
    # cv2.imwrite("result2.png", foveated_image)
