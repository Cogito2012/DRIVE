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


def padding_point(point, img_shape, shape_r=480, shape_c=640):
    """
    img_shape: [height, width]
    """
    def scale_point(point, img_shape, rows, cols):
        # compute the scale factor
        factor_scale_r = rows / img_shape[0]
        factor_scale_c = cols / img_shape[1]
        r = int(np.round(point[1] * factor_scale_r))
        c = int(np.round(point[0] * factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        return r, c
        
    rows_rate = img_shape[0] / shape_r
    cols_rate = img_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (img_shape[1] * shape_r) // img_shape[0]
        # scaling
        r, c = scale_point(point, img_shape, rows=shape_r, cols=new_cols)
        # shifting
        c = c + (shape_c - new_cols) // 2
    else:
        new_rows = (img_shape[0] * shape_c) // img_shape[1]
        # scaling
        r, c = scale_point(point, img_shape, rows=new_rows, cols=shape_c)
        # shifting
        r = r + (shape_r - new_rows) // 2
    new_point = np.array([c, r], dtype=np.int64)  # (x, y)
    return new_point


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


class ProcessFixations(object):
    """Pre-process fixation points to accord with the pre-processed images
    Args:
        input_shape: (shape_r, shape_c)
    """
    def __init__(self, input_shape, img_shape):
        if isinstance(input_shape, numbers.Number):
            self.input_shape = (int(input_shape), int(input_shape))
        else:
            self.input_shape = input_shape
        self.img_shape = img_shape

    def __call__(self, coords):
        """
        coords: fixation points, (L, 2) x, y
        """
        shape_r, shape_c = self.input_shape
        new_coords = np.zeros_like(coords, dtype=np.int64)
        for i, fixpt in enumerate(coords):
            if fixpt[0] > 0 and fixpt[1] > 0:
                new_coords[i] = padding_point(fixpt, self.img_shape, shape_r, shape_c)
        return new_coords

    def __repr__(self):
        return self.__class__.__name__ + '(input_shape={0})'.format(self.input_shape)