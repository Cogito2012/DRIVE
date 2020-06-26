
import cv2
import numpy as np
import math, time

class LaplacianBlending:
    def __init__(self,
                 _image,
                 _levels,
                 _kernels):
        self.image = _image
        self.levels = _levels
        self.kernels = _kernels

        self.imageLapPyr = []
        self.foveatedPyr = []
        self.image_sizes = []
        self.kernel_sizes = []

        self.imageSmallestLevel = []
        self.foveated_image = []

        self.buildPyramid()

        self.image_sizes = [None] * self.levels
        self.kernel_sizes = [None] * self.levels

        for i in range(self.levels-1, -1, -1):

            im_shape = np.shape(self.imageLapPyr[i])
            kernel_shape = np.shape(self.kernels[i])
            # shape=(width, height)
            self.image_sizes[i] = [im_shape[1], im_shape[0]]
            self.kernel_sizes[i] = [kernel_shape[1], kernel_shape[0]]


    def buildPyramid(self):
        self.imageLapPyr = []
        currentImg = self.image
        for i in range(0, self.levels):
            im_down = cv2.pyrDown(currentImg)
            im_up = cv2.pyrUp(im_down)
            # resize to avoid the odd scalar of the size of currentImg
            h, w, _ = np.shape(currentImg)
            im_up = cv2.resize(im_up, (w, h))
            im_lap = currentImg - im_up
            self.imageLapPyr.append(im_lap)
            currentImg = im_down

        self.imageSmallestLevel = im_up


    def foveate(self, fixation_point):

        self.foveated_image = self.imageSmallestLevel
        for i in range(self.levels-1, -1, -1):
            aux = []
            if i != 0:
                aux = fixation_point / np.power(2, i)
            else:
                aux = fixation_point

            upperleft_kernel = np.array(self.kernel_sizes[i]) / 2.0 - aux
            upperleft_kernel = upperleft_kernel.astype(int)
            x1 = upperleft_kernel[0]
            x2 = upperleft_kernel[0]+self.image_sizes[i][0]
            y1 = upperleft_kernel[1]
            y2 = upperleft_kernel[1]+self.image_sizes[i][1]
            subRect_kernel = self.kernels[i][y1:y2, x1:x2, :]

            aux_pyr = self.imageLapPyr[i] * subRect_kernel
            if i == self.levels-1:
                self.foveated_image = self.foveated_image + aux_pyr
            else:
                self.foveated_image = cv2.pyrUp(self.foveated_image)
                h, w, _ = np.shape(aux_pyr)
                self.foveated_image = cv2.resize(self.foveated_image, (w, h))
                self.foveated_image = self.foveated_image + aux_pyr

        return self.foveated_image


def createFilter(height, width, sigma):
    gKernel = np.zeros([height, width, 3], np.float64)
    s = 2.0 * sigma * sigma
    xc = int(round(width * 0.5))
    yc = int(round(height * 0.5))

    [ex, ey] = np.meshgrid(range(-xc, width - xc), range(-yc, height - yc))
    d = np.power(ex, 2) + np.power(ey, 2)
    gDist = np.exp(-d / s)
    maxValue = np.max(gDist)

    # normalize the kernel
    for i in range(3):
        gKernel[:, :, i] = gDist / maxValue

    return gKernel


def createFilterPyr(height, width, levels, sigma):

    kernels = []
    gKernel = createFilter(height, width, sigma)
    kernels.append(gKernel)

    for i in range(0,levels):
        kernel_down = cv2.pyrDown(kernels[i])
        kernels.append(kernel_down)

    return kernels


if __name__ == "__main__":

    image = cv2.imread('src/sample.png')
    fixation = np.array([450, 410])
    height, width, channel = np.shape(image)

    sigma = width/6.0  # param of gaussian kernel
    levels = 5   # number of gaussian pyramid levels
    factor = 2.0  # larger than 1.0

    image = np.float64(image)

    m = int(math.floor(factor * height))
    n = int(math.floor(factor * width))

    t_start = time.time()
    # compute kernels
    kernels = createFilterPyr(m, n, levels, sigma)

    # construct pyramid
    pyramid = LaplacianBlending(image, levels, kernels)

    fixation_point = [448, 86]
    foveated_image = pyramid.foveate(fixation_point)
    print("Ellapsed time: %.6f"%(time.time()-t_start))

    # cv2.imwrite("result.bmp", foveated_image)