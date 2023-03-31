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


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    # img = self.normalize(torch.from_numpy(img.copy()))
    img = torch.from_numpy(img.copy())
    return img


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_fps, img_size, buffer_size, info_prefix='info', ball_csv_prefix='ball_',
                 img_prefix='timestep_', **kwargs):
        self.data_fps = data_fps
        self.img_size = img_size
        self.buffer_size = buffer_size
        self.list_buffer_samples = []
        self.num_samples = []

        self.data_info_fps = []
        self.num_balls = []
        self.ball_csv_fpss = []
        for data_fp in data_fps:
            data_info_fp = os.path.join(data_fp, f'{info_prefix}.json')
            self.data_info_fps.append(data_info_fp)
            with open(data_info_fp) as f:
                data_info = json.load(f)

            num_balls = data_info['num_balls']
            ball_csv_fps = [os.path.join(data_fp, f"{ball_csv_prefix}{i}.csv") for i in range(num_balls)]

            self.num_balls.append(num_balls)
            self.ball_csv_fpss.append(ball_csv_fps)

            # parse the image files
            self.parse_input_list(data_fp, img_prefix, **kwargs)
        print(f'# buffer samples: {sum(self.num_samples) - self.buffer_size}')

        self.all_same_size = all(self.num_samples[0] == x for x in self.num_samples)
        self.idxs = list(range(sum(self.num_samples)))

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, data_fp, img_prefix, max_sample=-1, start_idx=-1, end_idx=-1):
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

        num_sample = len(list_sample) - self.buffer_size + 1
        assert num_sample > 0
        self.num_samples.append(num_sample)

        self.list_buffer_samples.append(
            [list_sample[i:i + self.buffer_size] for i in range(num_sample)])

    def idx2idxs(self, idx):
        """Convert from global index to dataset specific index"""
        if self.all_same_size:
            shape = [len(self.num_samples), self.num_samples[0]]
            num_dims = len(shape)
            offset = 1
            idxs = [0] * num_dims
            for i in range(num_dims - 1, -1, -1):
                idxs[i] = idx // offset % shape[i]
                offset *= shape[i]
        else:
            count = 0
            for i in range(len(self.num_samples)):
                if count + self.num_samples[i] > idx:
                    idxs = [i, idx - count]
                    break
                count += self.num_samples[i]
        return tuple(idxs)


class TrainDataset(BaseDataset):
    def __init__(self, data_fp, batch_size_per_gpu, img_size, buffer_size, **kwargs):
        super(TrainDataset, self).__init__(data_fp, img_size, buffer_size, **kwargs)
        self.batch_size_per_gpu = batch_size_per_gpu
        self.cur_idx = 0
        self.if_shuffled = False

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.idxs)
            self.if_shuffled = True

        batch_images = torch.zeros(self.batch_size_per_gpu, self.buffer_size, 3, self.img_size[0], self.img_size[1])
        batch_states = -1 * torch.ones(self.batch_size_per_gpu, self.buffer_size, max(self.num_balls),
                                       6)  # x, y, x_lin_vel, y_lin_vel, radius, label
        for bidx in range(self.batch_size_per_gpu):
            didx, tidx = self.idx2idxs(self.idxs[self.cur_idx])

            self.cur_idx += 1
            if self.cur_idx >= (sum(self.num_samples)):
                self.cur_idx = 0
                np.random.shuffle(self.idxs)

            batch_records = self.list_buffer_samples[didx][tidx]

            with open(self.data_info_fps[didx]) as f:
                data_info = json.load(f)
            ball_csvs = [pd.read_csv(fp, index_col=0) for fp in self.ball_csv_fpss[didx]]

            for i in range(self.buffer_size):
                record_img_fn = batch_records[i]

                # load image and label
                image_path = os.path.join(self.data_fps[didx], record_img_fn)
                img = Image.open(image_path).convert('RGB')
                if not ((img.width == self.img_size[0]) and (img.height == self.img_size[1])):
                    img = img_resize(img, (self.img_size[0], self.img_size[1]), interp='bilinear')
                # image transform, to torch float tensor 3xHxW
                img = img_transform(img)

                # put into batch arrays
                batch_images[bidx, i][:, :img.shape[1], :img.shape[2]] = img

                record_idx = extract_integer(record_img_fn)
                for j in range(self.num_balls[didx]):
                    record_state = list(ball_csvs[j].iloc[record_idx][['pose_x', 'pose_y', 'vel_lin_x', 'vel_lin_y']]) \
                                   + [data_info[str(j)]['radius']] + [data_info[str(j)]['label']]
                    batch_states[bidx, i][j] = torch.FloatTensor(record_state)

        output = dict()
        output['img_data'] = batch_images
        output['state_label'] = batch_states
        return output

    def __len__(self):
        return int(1e10)
        # return sum(self.num_samples)
