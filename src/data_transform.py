""" This file is modified from:
https://raw.githubusercontent.com/piergiaj/pytorch-i3d/master/videotransforms.py
"""
import numpy as np
import cv2
import numbers
import random


def padding_inv(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img / (np.max(img) + 1e-6) * 255


def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


class ProcessImages(object):
    """Pre-process images with padded resize operation, and normalize
    Args:
        input_shape: (shape_r, shape_c)
    """
    def __init__(self, input_shape):
        if isinstance(input_shape, numbers.Number):
            self.input_shape = (int(input_shape), int(input_shape))
        else:
            self.input_shape = input_shape

    def __call__(self, imgs):
        """
        imgs: RGB images (T, H, W, C)
        """
        t, h, w, c = imgs.shape
        shape_r, shape_c = self.input_shape
        
        ims = np.zeros((t, shape_r, shape_c, c), dtype=np.float64)
        for i, im in enumerate(imgs):
            padded_image = padding(im, shape_r, shape_c, c)
            if c == 1:
                padded_image = np.expand_dims(padded_image, axis=-1)
            ims[i] = padded_image.astype(np.float64)
        # normalize
        ims /= 255.0
        ims = np.rollaxis(ims, 3, 1)  # (t, c, h, w)
        return ims

    def __repr__(self):
        return self.__class__.__name__ + '(input_shape={0})'.format(self.input_shape)