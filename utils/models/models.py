# -*- coding: utf-8 -*-
# This file is code altered from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet
from utils.optical_flow import spynet_optical_flow, farneback_optical_flow
from utils.mask import tensor_arr_dist_circle_mask, tensor_img_dist_circle_mask

# default input dimensions
DEFAULT_RGB_DIM = 3
# default optical flow dimensions
DEFAULT_OPTICAL_FLOW_DIM = 2
# default prediction classes (ImageNet)
DEFAULT_NUM_CLASSES = 1000


class PerceptualModuleBase(nn.Module):
    def __init__(self):
        super(PerceptualModuleBase, self).__init__()

    def label_pred(self, output):
        pred_log_prob = F.log_softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        return pred

    def label_acc(self, pred, label):
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (pred == label).long())
        valid_sum = torch.sum(valid)
        acc = acc_sum.float() / (valid_sum.float() + 1e-10)
        return acc


class PerceptualModule(PerceptualModuleBase):
    def __init__(self, optical_flow, net_perceptual, crit, buffer_size):
        super(PerceptualModule, self).__init__()
        self.optical_flow = optical_flow
        self.net_perceptual = net_perceptual
        self.crit = crit
        self.buffer_size = buffer_size

    def process_feed(self, feed_dict):
        input_data = feed_dict['img_data']
        output_data = feed_dict['state_label']

        BA, BU, _, H, W = input_data.shape
        _, _, max_num_balls, num_features = output_data.shape
        assert BU == self.buffer_size
        C_Final = BU * 3 + (BU - 1) * 2

        input_processed = torch.zeros(BA, max_num_balls, C_Final, H, W)
        output_processed = torch.zeros(BA, max_num_balls, dtype=torch.long)
        for i in range(BA):
            for j in range(BU):
                img_orig = input_data[i, j, ...]
                for k in range(max_num_balls):
                    cx, cy, cr = output_data[i, j, k, [0, 1, -2]]
                    img_masked = tensor_img_dist_circle_mask(img_orig, cx, cy, cr + 0.05)
                    input_processed[i, k, j * 3:(j + 1) * 3, ...] = img_masked

                    if j != BU - 1:
                        opt_flow = self.optical_flow(input_data[i][j], input_data[i][j + 1])
                        opt_flow_masked = tensor_arr_dist_circle_mask(opt_flow, cx, cy, cr, 0.1)
                        input_processed[i, k, BU * 3 + j * 2:BU * 3 + (j + 1) * 2, ...] = opt_flow_masked

                    output_processed[i, k] = output_data[i, j, k, -1]

        input_processed = torch.flatten(input_processed, end_dim=1).cuda()
        output_processed = torch.flatten(output_processed, end_dim=1).cuda()
        return input_processed, output_processed

    def forward(self, feed):
        if isinstance(feed, list):
            feed = feed[0]

        samples, labels = self.process_feed(feed)
        output = self.net_perceptual(samples)
        prediction = self.label_pred(output)

        if self.training:
            loss = self.crit(output, labels)
            acc = self.label_acc(prediction, labels)
            return loss, acc
        else:
            return prediction


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def build_perceptual(arch='resnet18', input_dim=3, num_classes=1000, weights=''):
        pretrained = len(weights) == 0 and (input_dim == DEFAULT_RGB_DIM) and (num_classes == DEFAULT_NUM_CLASSES)
        arch = arch.lower()
        if arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](input_dim=input_dim, num_classes=num_classes,
                                                      pretrained=pretrained)
            net_perceptual = Resnet(orig_resnet)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](input_dim=input_dim, num_classes=num_classes,
                                                      pretrained=pretrained)
            net_perceptual = Resnet(orig_resnet)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_perceptual')
            net_perceptual.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        else:
            net_perceptual.apply(ModelBuilder.weights_init)
        return net_perceptual

    @staticmethod
    def build_optical_flow(arch='spynet'):
        arch = arch.lower()
        if arch == 'spynet':
            optical_flow = spynet_optical_flow
        elif arch == 'farneback':
            optical_flow = farneback_optical_flow
        else:
            raise Exception('Architecture undefined!')
        return optical_flow


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        self.avgpool = orig_resnet.avgpool
        self.fc = orig_resnet.fc

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
