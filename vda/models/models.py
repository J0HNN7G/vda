# -*- coding: utf-8 -*-
# This file is code altered from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet
from vda.optical_flow import spynet_optical_flow, farneback_optical_flow
from vda.pose import get_circular_poses
from vda.mask import tensor_arr_dist_circle_mask, tensor_img_dist_circle_mask, tensor_img_px_circle_mask, \
    get_tensor_img_px_circle_mask_rgb

from vda.visualization import show_farneback_optical_flows

# default input dimensions
DEFAULT_RGB_DIM = 3
# default optical flow dimensions
DEFAULT_OPTICAL_FLOW_DIM = 2
# default prediction classes (ImageNet)
DEFAULT_NUM_CLASSES = 1000
# offsets for masking
DEFAULT_MASK_PX_OFFSET = 2
DEFAULT_MASK_DIST_OFFSET = 0.05


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
    def __init__(self, optical_flow, net_perceptual, crit, buffer_size, include_images, include_optical_flow):
        super(PerceptualModule, self).__init__()
        assert (include_images or include_optical_flow)
        self.optical_flow = optical_flow
        self.net_perceptual = net_perceptual
        self.crit = crit
        self.buffer_size = buffer_size
        self.include_images = include_images
        self.include_optical_flow = include_optical_flow
        self.C_Final = self.include_images * self.buffer_size * 3 + self.include_optical_flow * (
                self.buffer_size - 1) * 2

    def training_process_feed(self, feed_dict):
        input_data = feed_dict['img_data']
        output_data = feed_dict['state_label']

        BA, BU, _, H, W = input_data.shape
        _, _, max_num_balls, num_features = output_data.shape
        assert BU == self.buffer_size

        input_processed = torch.zeros(BA, max_num_balls, self.C_Final, H, W)
        output_processed = torch.zeros(BA, max_num_balls, dtype=torch.long)
        for i in range(BA):
            for j in range(BU):
                img_orig = input_data[i, j, ...]
                for k in range(max_num_balls):
                    cx, cy, cr = output_data[i, j, k, [0, 1, -2]]
                    if self.include_images:
                        img_masked = tensor_img_dist_circle_mask(img_orig, cx, cy, cr + DEFAULT_MASK_DIST_OFFSET)
                        input_processed[i, k, j * 3:(j + 1) * 3, ...] = img_masked

                    if self.include_optical_flow and (j != BU - 1):
                        opt_flow = self.optical_flow(input_data[i][j], input_data[i][j + 1])
                        opt_flow_masked = tensor_arr_dist_circle_mask(opt_flow, cx, cy, cr)

                        c_start_idx = self.include_images * BU * 3 + j * 2
                        c_end_idx = self.include_images * BU * 3 + (j + 1) * 2
                        input_processed[i, k, c_start_idx:c_end_idx, ...] = opt_flow_masked

                    output_processed[i, k] = output_data[i, j, k, -1]

        input_processed = torch.flatten(input_processed, end_dim=1).cuda()
        output_processed = torch.flatten(output_processed, end_dim=1).cuda()
        return input_processed, output_processed

    def eval_process_feed2(self, feed_dict):
        input_data = feed_dict['img_data']

        BA, BU, _, H, W = input_data.shape
        assert BA == 1  # not implemented for more
        assert BU == self.buffer_size

        poses = get_circular_poses(input_data[0, 0, ...].cpu())
        num_balls = len(poses)  # assume number of balls from first frame

        input_processed = torch.zeros(1, num_balls, self.C_Final, H, W)
        intrinsic_parameters = torch.zeros(1, num_balls, 6, dtype=torch.float)  # r, g, b, radius, mass, friction
        extrinsic_parameters = torch.zeros(1, num_balls, 4, dtype=torch.float)
        for j in range(BU):
            img_orig = input_data[0, j].cpu()
            for k in range(num_balls):
                px, py, pr = poses[k].numpy()

                if j == 0:
                    mean_rgb = get_tensor_img_px_circle_mask_rgb(img_orig, px, py, pr).mean(axis=0).numpy()
                    intrinsic_parameters[0, k, :] = torch.FloatTensor(
                        [mean_rgb[0], mean_rgb[1], mean_rgb[2], pr, -1, -1])

                if self.include_images:
                    img_masked = tensor_img_px_circle_mask(img_orig, px, py, pr + DEFAULT_MASK_PX_OFFSET)
                    input_processed[0, k, j * 3:(j + 1) * 3, ...] = img_masked

                if self.include_optical_flow and (j != BU - 1):
                    opt_flow = self.optical_flow(img_orig, input_data[0, j + 1].cpu())
                    opt_flow_masked = tensor_img_px_circle_mask(opt_flow, px, py, pr)

                    c_idx_start = self.include_images * BU * 3 + j * 2
                    c_idx_end = self.include_images * BU * 3 + (j + 1) * 2
                    input_processed[0, k, c_idx_start:c_idx_end, ...] = opt_flow_masked

                # updating poses
                vx, vy = opt_flow[:, int(py), int(px)].numpy()
                poses[k, ...] = torch.FloatTensor([px + vx, py + vy, pr])
                if j == BU - 1:
                    extrinsic_parameters[0, k, :] = torch.FloatTensor([px, py, vx, vy])

        input_processed = torch.flatten(input_processed, end_dim=1).cuda()
        intrinsic_parameters = torch.flatten(intrinsic_parameters, end_dim=1)
        extrinsic_parameters = torch.flatten(extrinsic_parameters, end_dim=1)
        return input_processed, intrinsic_parameters, extrinsic_parameters

    def eval_process_feed(self, feed_dict):
        input_data = feed_dict['img_data']

        BA, BU, _, H, W = input_data.shape
        assert BA == 1  # not implemented for more
        assert BU == self.buffer_size

        poses = get_circular_poses(input_data[0, -1, ...].cpu())
        num_balls = len(poses)  # assume number of balls from first frame

        input_processed = torch.zeros(1, num_balls, self.C_Final, H, W)
        intrinsic_parameters = torch.zeros(1, num_balls, 6, dtype=torch.float)  # r, g, b, radius, mass, friction
        extrinsic_parameters = torch.zeros(1, num_balls, 4, dtype=torch.float)
        for j in range(1, BU)[::-1]:
            img_orig = input_data[0, j].cpu()
            for k in range(num_balls):
                px, py, pr = poses[k].numpy()

                if self.include_images:
                    img_masked = tensor_img_px_circle_mask(img_orig, px, py, pr + DEFAULT_MASK_PX_OFFSET)
                    input_processed[0, k, (j-1) * 3:j * 3, ...] = img_masked

                if j != 0:
                    opt_flow_forward = self.optical_flow(input_data[0, j - 1].cpu(), img_orig)
                    opt_flow_back = self.optical_flow(img_orig, input_data[0, j - 1].cpu())

                    rev_vx, rev_vy = opt_flow_back[:, int(py), int(px)].numpy()
                    prev_px = px + rev_vx
                    prev_py = py + rev_vy

                    if self.include_optical_flow:
                        opt_flow_masked = tensor_img_px_circle_mask(opt_flow_forward, prev_px, prev_py, pr)

                        c_idx_start = self.include_images * BU * 3 + (j - 1) * 2
                        c_idx_end = self.include_images * BU * 3 + j * 2
                        input_processed[0, k, c_idx_start:c_idx_end, ...] = opt_flow_masked

                    if j == BU - 1:
                        extrinsic_parameters[0, k, :] = torch.FloatTensor([px, py, - rev_vx, - rev_vy])

                        # intrinsic
                        mean_rgb = get_tensor_img_px_circle_mask_rgb(img_orig, px, py, pr).mean(axis=0).numpy()
                        intrinsic_parameters[0, k, :] = torch.FloatTensor(
                            [mean_rgb[0], mean_rgb[1], mean_rgb[2], pr, -1, -1])

                    poses[k, ...] = torch.FloatTensor([prev_px, prev_py, pr])

        input_processed = torch.flatten(input_processed, end_dim=1).cuda()
        intrinsic_parameters = torch.flatten(intrinsic_parameters, end_dim=1)
        extrinsic_parameters = torch.flatten(extrinsic_parameters, end_dim=1)
        return input_processed, intrinsic_parameters, extrinsic_parameters

    def forward(self, feed):
        if isinstance(feed, list):
            feed = feed[0]

        if self.training:
            samples, labels = self.training_process_feed(feed)
            output = self.net_perceptual(samples)
            prediction = self.label_pred(output)

            loss = self.crit(output, labels)
            acc = self.label_acc(prediction, labels)
            return loss, acc
        else:
            samples, intrinsic_params, extrinsic_params = self.eval_process_feed(feed)
            output = self.net_perceptual(samples)
            predicted_params = self.label_pred(output)

            return predicted_params, intrinsic_params, extrinsic_params


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
    def build_optical_flow(arch='farneback'):
        arch = arch.lower()
        if arch == 'farneback':
            optical_flow = farneback_optical_flow
        elif arch == 'spynet':
            optical_flow = spynet_optical_flow
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
