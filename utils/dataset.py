# -*- coding: utf-8 -*-
# This file is code altered from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

import os
import json
import re

import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


def img_resize(img, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return img.resize(size, resample)


def extract_integer(filename):
    return int(filename.split('.')[0].split('_')[1])


def createSingleTensorDataset(data_fp, img_size, buffer_size, info_prefix='info', ball_csv_prefix='ball_',
                              img_prefix='timestep_'):
    data_info_fp = os.path.join(data_fp, f'{info_prefix}.json')
    with open(data_info_fp) as f:
        data_info = json.load(f)
    num_balls = data_info['num_balls']

    ball_csv_fps = [os.path.join(data_fp, f"{ball_csv_prefix}{i}.csv") for i in range(num_balls)]
    ball_csvs = [pd.read_csv(fp, index_col=0) for fp in ball_csv_fps]

    list_sample = []
    fp_filter = re.compile(fr'{img_prefix}\d+.png')
    img_fps = [fp for fp in os.listdir(data_fp) if re.match(fp_filter, fp)]
    for fp in sorted(img_fps, key=extract_integer):
        list_sample.append(fp)

    list_buffer_sample_len = len(list_sample) - buffer_size + 1
    list_buffer_sample = [list_sample[i:i + buffer_size] for i in range(list_buffer_sample_len)]

    xs = torch.zeros(list_buffer_sample_len, buffer_size, 3, img_size[0], img_size[1])

    batch_states = torch.zeros(list_buffer_sample_len, buffer_size, num_balls, 6)  # x, y, x_lin_vel, y_lin_vel, radius, label
    for i in range(buffer_size):
        record_img_fn = batch_records[i]

        # load image and label
        image_path = os.path.join(data_fp, record_img_fn)
        img = Image.open(image_path).convert('RGB')
        if not ((img.width == img_size[0]) and (img.height == img_size[1])):
            img = img_resize(img, (img_size[0], img_size[1]), interp='bilinear')
        # image transform, to torch float tensor 3xHxW
        img = img_transform(img)

        # put into batch arrays
        batch_images[i][:, :img.shape[1], :img.shape[2]] = img

        record_idx = extract_integer(record_img_fn)
        for j in range(num_balls):
            record_state = list(ball_csvs[j].iloc[record_idx][['pose_x', 'pose_y', 'vel_lin_x', 'vel_lin_y']]) \
                           + [data_info[str(j)]['radius']] + [data_info[str(j)]['label']]
            batch_states[i][j] = torch.FloatTensor(record_state)

    output = dict()
    output['img_data'] = batch_images
    output['state_label'] = batch_states

    return -1


def parse_input_list(data_fp, img_prefix, buffer_size, max_sample=-1, start_idx=-1, end_idx=-1):
    if not os.path.exists(data_fp):
        raise NotADirectoryError
    list_sample = []
    fp_filter = re.compile(fr'{img_prefix}\d+.png')
    img_fps = [fp for fp in os.listdir(data_fp) if re.match(fp_filter, fp)]
    for fp in sorted(img_fps, key=extract_integer):
        list_sample.append(fp)

    if max_sample > 0:
        list_sample = list_sample[0:max_sample]
    if start_idx >= 0 and end_idx >= 0:  # divide file list
        list_sample = list_sample[start_idx:end_idx]

    num_sample = len(list_sample)
    assert num_sample > 0

    list_buffer_sample = [list_sample[i:i + buffer_size] for i in range(num_sample - buffer_size + 1)]

    return list_buffer_sample


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy())
    return img


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_fp, img_size, buffer_size, info_prefix='info', ball_csv_prefix='ball_',
                 img_prefix='timestep_', **kwargs):
        self.data_fp = data_fp
        self.img_size = img_size
        self.buffer_size = buffer_size
        self.list_sample = None
        self.list_buffer_sample = None
        self.num_sample = -1

        self.data_info_fp = os.path.join(data_fp, f'{info_prefix}.json')
        with open(self.data_info_fp) as f:
            data_info = json.load(f)

        self.num_balls = data_info['num_balls']
        self.ball_csv_fps = [os.path.join(data_fp, f"{ball_csv_prefix}{i}.csv") for i in range(self.num_balls)]

        # parse the image files
        self.parse_input_list(data_fp, img_prefix, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, data_fp, img_prefix, max_sample=-1, start_idx=-1, end_idx=-1):
        if not os.path.exists(data_fp):
            raise NotADirectoryError
        self.list_sample = []
        fp_filter = re.compile(fr'{img_prefix}\d+.png')
        img_fps = [fp for fp in os.listdir(data_fp) if re.match(fp_filter, fp)]
        for fp in sorted(img_fps, key=extract_integer):
            self.list_sample.append(fp)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:  # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0

        self.list_buffer_sample = [self.list_sample[i:i + self.buffer_size]
                                   for i in range(self.num_sample - self.buffer_size + 1)]

        # print('# individual frames: {}'.format(self.num_sample))
        # print('# buffer samples: {}'.format(self.num_sample - self.buffer_size))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        # img = self.normalize(torch.from_numpy(img.copy()))
        img = torch.from_numpy(img.copy())
        return img


class TrainSingleDataset(BaseDataset):
    def __init__(self, data_fp, img_size, buffer_size, **kwargs):
        super(TrainSingleDataset, self).__init__(data_fp, img_size, buffer_size, **kwargs)
        self.cur_idx = 0
        self.if_shuffled = False

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_buffer_sample)
            self.if_shuffled = True

        batch_records = self.list_buffer_sample[self.cur_idx]
        self.cur_idx += 1
        if self.cur_idx >= (self.num_sample - self.buffer_size):
            self.cur_idx = 0
            np.random.shuffle(self.list_sample)

        with open(self.data_info_fp) as f:
            data_info = json.load(f)
        ball_csvs = [pd.read_csv(fp, index_col=0) for fp in self.ball_csv_fps]

        batch_images = torch.zeros(self.buffer_size, 3, self.img_size[0], self.img_size[1])
        batch_states = torch.zeros(self.buffer_size, self.num_balls, 6)  # x, y, x_lin_vel, y_lin_vel, radius, label
        for i in range(self.buffer_size):
            record_img_fn = batch_records[i]

            # load image and label
            image_path = os.path.join(self.data_fp, record_img_fn)
            img = Image.open(image_path).convert('RGB')
            if not ((img.width == self.img_size[0]) and (img.height == self.img_size[1])):
                img = img_resize(img, (self.img_size[0], self.img_size[1]), interp='bilinear')
            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img

            record_idx = extract_integer(record_img_fn)
            for j in range(self.num_balls):
                record_state = list(ball_csvs[j].iloc[record_idx][['pose_x', 'pose_y', 'vel_lin_x', 'vel_lin_y']]) \
                               + [data_info[str(j)]['radius']] + [data_info[str(j)]['label']]
                batch_states[i][j] = torch.FloatTensor(record_state)

        output = dict()
        output['img_data'] = batch_images
        output['state_label'] = batch_states
        return output

    def __len__(self):
        return self.num_sample - self.buffer_size


class TrainMultiDataset(torch.utils.data.Dataset):
    def __init__(self, data_fps, img_size, buffer_size, random_order=False, **kwargs):
        self.datasets = []
        for data_fp in data_fps:
            self.datasets.append(TrainSingleDataset(data_fp, img_size, buffer_size))
        self.dataset_cur_idx = -1
        self.random_order = random_order
        self.if_shuffled = False

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if self.random_order:
            dataset = self.datasets[np.random.randint(len(self.datasets))]
        else:
            if not self.if_shuffled:
                self.dataset_cur_idx = 0
                np.random.shuffle(self.datasets)
                self.if_shuffled = True

            dataset = self.datasets[self.dataset_cur_idx]

            if dataset.cur_idx == len(dataset):
                self.dataset_cur_idx += 1
            if self.dataset_cur_idx > len(self.datasets):
                np.random.shuffle(self.datasets)
        return dataset[-1]

    def __len__(self):
        # return int(1e10)  # It's a fake length due to the trick that every loader maintains its own list
        return sum([len(dataset) for dataset in self.datasets])
