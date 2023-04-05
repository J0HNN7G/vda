# -*- coding: utf-8 -*-
# This file is code from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# system libs
import sys
import logging
# numerical libs
import math
import torch


def setup_logger(distributed_rank=0, filename="log.txt"):
    """
    Set up a logger to write log messages to a file and/or console.

    Args:
        distributed_rank (int): The rank of the current process in a distributed environment. If it is greater than
            zero, log messages will not be written to the console. Default is 0.
        filename (str): The name of the file to write log messages to. Default is "log.txt".

    Returns:
        logging.Logger: The logger object that can be used to write log messages.
    """
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger


class MetricMeter(object):
    """
    Computes and stores the average, current value, and standard deviation of a metric.

    Attributes:
        initialized (bool): True if the meter has been initialized, False otherwise.
        val (float or None): The current value of the metric.
        avg (float or None): The average value of the metric.
        sum (float or None): The sum of the metric values.
        count (float or None): The number of times the metric has been updated.
        std (float or None): The standard deviation of the metric.
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.std = None

    def initialize(self, val, weight):
        """
        Initialize the meter with an initial value and weight.

        Args:
            val (float): The initial value of the metric.
            weight (float): The weight of the initial value.

        Returns:
            None
        """
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.std = 0
        self.initialized = True

    def update(self, val, weight=1):
        """
        Update the meter with a new value and weight.

        Args:
            val (float): The new value of the metric.
            weight (float): The weight of the new value. Default is 1.

        Returns:
            None
        """
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        """
        Add a new value and weight to the meter.

        Args:
            val (float): The new value to add.
            weight (float): The weight of the new value.

        Returns:
            None
        """
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
        self.std = math.sqrt(((self.count - 1) * self.std ** 2 + weight * (val - self.avg) ** 2) / self.count)

    def value(self):
        """
        Get the current value of the meter.

        Returns:
            float or None: The current value of the meter.
        """
        return self.val

    def average(self):
        """
        Get the average value of the meter.

        Returns:
            float or None: The average value of the meter.
        """
        return self.avg

    def standard_deviation(self):
        """
        Get the standard deviation of the meter.

        Returns:
            float or None: The standard deviation of the meter.
        """
        return self.std


def label_accuracy(prediction, label):
    """
    Compute the accuracy of predicted labels compared to ground truth labels.

    Args:
        prediction (torch.Tensor): A PyTorch tensor representing the predicted labels.
        label (torch.Tensor): A PyTorch tensor representing the ground truth labels.

    Returns:
        float: The accuracy of the predicted labels compared to the ground truth labels.
    """
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (prediction == label).long())
    valid_sum = torch.sum(valid)
    acc = acc_sum.float() / (valid_sum.float() + 1e-10)
    return acc


def manhattan_distance(x, y):
    """
    Compute the Manhattan distance between two vectors x and y using PyTorch.

    Args:
        x (torch.Tensor): A PyTorch tensor of shape (N, D) representing the first vector.
        y (torch.Tensor): A PyTorch tensor of shape (N, D) representing the second vector.

    Returns:
        torch.Tensor: A PyTorch tensor of shape (N,) containing the Manhattan distance between each pair of
        vectors in x and y.
    """
    return torch.sum(torch.abs(x - y), dim=1)


def pixel_mse(image1, image2):
    """
    Compute the pixel-wise mean squared error (MSE) between two images with dimensions C,H,W using PyTorch.

    Args:
        image1 (torch.Tensor): A PyTorch tensor of shape (C, H, W) representing the first image.
        image2 (torch.Tensor): A PyTorch tensor of shape (C, H, W) representing the second image.

    Returns:
        float: The pixel-wise mean squared error (MSE) between image1 and image2.
    """
    return torch.mean(torch.pow(image1 - image2, 2))


def calculate_circle_iou(circle1, circle2):
    """
    Calculate the Intersection over Union (IOU) between two circular regions.

    Args:
        circle1 (list or tensor): The (x, y) coordinates of the center of the first circle and its radius.
        circle2 (list or tensor): The (x, y) coordinates of the center of the second circle and its radius.

    Returns:
        The IOU between the two circles as a float or tensor.
    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance = torch.sqrt(torch.pow(x2 - x1, 2) + torch.pow(y2 - y1, 2))
    if distance > r1 + r2:
        return torch.tensor(0.0)
    elif distance <= torch.abs(r1 - r2):
        return torch.tensor(1.0)
    else:
        a = r1 ** 2 * torch.acos((distance ** 2 + r1 ** 2 - r2 ** 2) / (2 * distance * r1))
        b = r2 ** 2 * torch.acos((distance ** 2 + r2 ** 2 - r1 ** 2) / (2 * distance * r2))
        c = 0.5 * ((-distance + r1 + r2) * (distance + r1 - r2) * (distance - r1 + r2) * (distance + r1 + r2)) ** 0.5
        intersection = a + b - c
        union = math.pi * (r1 ** 2 + r2 ** 2) - intersection
        return intersection / union


def calculate_object_proposal_metrics(prediction, ground_truth, iou_cutoff=0.8):
    """
    Calculate the accuracy, recall, precision, F1 score, and mean IOU for a list of circular region proposals.

    Args:
        prediction (list or tensor): The (x, y) coordinates of the centers of the predicted circles and their radii.
        ground_truth (list or tensor): The (x, y) coordinates of the centers of the ground truth circles and their radii.
        iou_cutoff (float): The IOU cutoff value above which a prediction is considered a true positive.

    Returns:
        A tuple containing the accuracy, recall, precision, F1 score, mean IOU as floats or tensors, and a list of integers indicating
        which circle in the prediction list is a true positive for each circle in the ground truth list.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_positive_indices = []
    iou_list = []
    used_indices = set()
    for i, circle1 in enumerate(ground_truth):
        iou_list_per_circle = []
        index_list_per_circle = []
        for j, circle2 in enumerate(prediction):
            if j not in used_indices:
                iou = calculate_circle_iou(circle1, circle2)
                iou_list_per_circle.append(iou)
                index_list_per_circle.append(j)
        if iou_list_per_circle:
            max_iou, max_index = torch.max(torch.stack(iou_list_per_circle)), torch.argmax(torch.stack(iou_list_per_circle))
            if max_iou > iou_cutoff:
                true_positives += 1
                true_positive_indices.append(index_list_per_circle[max_index])
                used_indices.add(index_list_per_circle[max_index])
                iou_list.append(max_iou)
            else:
                false_negatives += 1
                true_positive_indices.append(None)
        else:
            false_negatives += 1
            true_positive_indices.append(None)
    for i, circle1 in enumerate(prediction):
        if i not in used_indices:
            iou_list_per_circle = []
            for circle2 in ground_truth:
                iou = calculate_circle_iou(circle1, circle2)
                iou_list_per_circle.append(iou)
            max_iou = torch.max(torch.stack(iou_list_per_circle))
            if max_iou <= iou_cutoff:
                false_positives += 1
    total_objects = true_positives + false_positives + false_negatives
    accuracy = true_positives / total_objects if total_objects > 0 else 0
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    mean_iou = torch.mean(torch.stack(iou_list)) if iou_list else 0
    return accuracy, recall, precision, f1_score, mean_iou, true_positive_indices
