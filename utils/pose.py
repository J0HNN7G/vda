# -*- coding: utf-8 -*-

import cv2
import torch


def tensor_to_cv2(img):
    return (img.permute(1, 2, 0) * 255.).type(torch.uint8).numpy()


def get_circular_poses(img):
    img = tensor_to_cv2(img)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.1
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)

    poses = torch.zeros(len(keypoints), 3)
    for i, keypoint in enumerate(keypoints):
        poses[i, ...] = torch.FloatTensor([keypoint.pt[0], keypoint.pt[1], keypoint.size / 2])
    return poses