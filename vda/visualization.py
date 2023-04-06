# -*- coding: utf-8 -*-

import torchvision.transforms.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_imgs(imgs):
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show_farneback_optical_flows(opt_flows):
    fig, axs = plt.subplots(ncols=len(opt_flows), squeeze=False)
    for i, opt_flow in enumerate(opt_flows):
        opt_flow = opt_flow.detach().numpy()
        hsv = np.zeros((256, 256, 3), dtype=np.float32)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(opt_flow[0, ...], opt_flow[1, ...])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        axs[0, i].imshow(np.asarray(bgr))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def save_spynet_optical_flows(opt_flows):
    for i, opt_flow in enumerate(opt_flows):
        objOutput = open(f'out_{i}.flo', 'wb')
        np.array([80, 73, 69, 72], np.uint8).tofile(objOutput)
        np.array([opt_flow.shape[2], opt_flow.shape[1]], np.int32).tofile(objOutput)
        np.array(opt_flow.detach().numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)
        objOutput.close()
