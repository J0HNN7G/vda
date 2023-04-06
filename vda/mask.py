# -*- coding: utf-8 -*-
import torch
import numpy as np


def scale2scale(value, oMin=-1.0, oMax=1.0, nMin=-1.0, nMax=1.0):
    """
    Convert linear scale (min/max) to another linear scale (min/max)
    value: value to be converted
    oMin: old minimum value
    oMax: old maximum value
    nMin: new minimum value
    nMax: new maximum value
    return: value mapped from old range to new range
    """
    oSpan = oMax - oMin
    nSpan = nMax - nMin
    result = ((value - oMin) / oSpan) * nSpan + nMin
    return result


def bool_circle_mask(arr, xidx, yidx, rad):
    c, h, w = arr.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - xidx) ** 2 + (Y - yidx) ** 2)
    mask = dist_from_center <= rad
    mask = np.repeat(mask[np.newaxis, :, :], c, axis=0)
    return mask


def arr_circle_mask(arr, xidx, yidx, rad, default_val=0):
    result = np.ones_like(arr) * default_val
    mask = bool_circle_mask(arr, xidx, yidx, rad)
    result[mask] = arr[mask]
    return result


def get_tensor_img_px_circle_mask_rgb(img, px, py, pr):
    mask = torch.tensor(bool_circle_mask(img.detach().numpy(), px, py, pr))
    masked_values = img[mask].view(3, -1)
    rgb_values = masked_values.t()
    return rgb_values


def tensor_arr_dist_circle_mask(arr, cx, cy, cr, extra_pad=0.05):
    arr = arr.detach().numpy()
    xidx = int(scale2scale(cx, -1.0, 1.0, 0.0, 256.0))
    yidx = int(scale2scale(cy, -1.0, 1.0, 256.0, 0.0))
    rad = int(scale2scale(cr + extra_pad, 0.0, 1.0, 0.0, 256.0 // 2))
    result = arr_circle_mask(arr, xidx, yidx, rad)
    result = torch.from_numpy(result.copy())
    return result


def tensor_img_px_circle_mask(img, px, py, pr):
    result = arr_circle_mask(img.detach().numpy(), px, py, pr)
    result = torch.from_numpy(result.copy())
    return result


def tensor_img_dist_circle_mask(img, cx, cy, cr, extra_pad=0.05):
    px = int(scale2scale(cx, -1.0, 1.0, 0.0, 256.0))
    py = int(scale2scale(cy, -1.0, 1.0, 256.0, 0.0))
    pr = int(scale2scale(cr + extra_pad, 0.0, 1.0, 0.0, 256.0 // 2))
    return tensor_img_px_circle_mask(img, px, py, pr)
