# -*- coding: utf-8 -*-
import cv2
import torch

from utils.pose import tensor_to_cv2


def farneback_optical_flow(img1, img2):
    # pre-processing
    img1 = cv2.cvtColor(tensor_to_cv2(img1), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(tensor_to_cv2(img2), cv2.COLOR_RGB2GRAY)

    opt_flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    opt_flow = opt_flow.transpose((2, 1, 0))
    opt_flow = torch.from_numpy(opt_flow.copy())
    return opt_flow
