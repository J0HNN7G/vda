# -*- coding: utf-8 -*-
# This file is code altered from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

import os
import json
import re
# numerical libs
import torch
from PIL import Image
import pandas as pd
import numpy as np

from utils.mask import scale2scale


TEST_BUFFER_PREFIX = 'buffer'
TEST_TIMESTEP_PREFIX = 'timestep'


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


def extract_integers(filename):
    split_list = filename.split('.')[0].split('_')
    int1 = split_list[1]
    int2 = split_list[3]
    return int(f'{int1}{int2}')


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
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

        self.all_same_size = all(self.num_samples[0] == x for x in self.num_samples)
        self.idxs = list(range(sum(self.num_samples)))

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
    def __init__(self, data_fp, cfg, **kwargs):
        super(TrainDataset, self).__init__(data_fp, cfg.DATASET.img_size, cfg.DATASET.buffer_size, **kwargs)
        self.batch_size_per_gpu = cfg.TRAIN.batch_size_per_gpu
        self.cur_idx = 0
        self.if_shuffled = False
        print(f'# buffer samples: {sum(self.num_samples) - self.buffer_size}')

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


class ValDataset(BaseDataset):
    def __init__(self, data_fp, cfg, **kwargs):
        super(ValDataset, self).__init__(data_fp, cfg.DATASET.img_size, cfg.DATASET.buffer_size, **kwargs)
        self.cur_idx = -1
        self.cfg = cfg
        print(f'# samples: {len(self)}')

    def __getitem__(self, index):
        batch_images = torch.zeros(1, self.buffer_size, 3, self.img_size[0], self.img_size[1])
        batch_labels = -1 * torch.ones(1, max(self.num_balls))
        batch_prop_states = -1 * torch.ones(1, max(self.num_balls), 3) # x,y,radius
        batch_ext_states = -1 * torch.ones(1, len(self.cfg.VAL.prediction_timesteps), max(self.num_balls),
                                           4)  # x, y, x_lin_vel, y_lin_vel

        self.cur_idx += 1
        if self.cur_idx >= (sum(self.num_samples)):
            self.cur_idx = 0
        didx, tidx = self.idx2idxs(self.idxs[self.cur_idx])
        orig_idx = self.cur_idx

        while self.num_samples[didx] - 1 - tidx < max(self.cfg.VAL.prediction_timesteps):
            self.cur_idx += self.num_samples[didx] - tidx

            if self.cur_idx >= (sum(self.num_samples)):
                raise RuntimeError('Dataset too small to evaluate future timesteps')

            didx, tidx = self.idx2idxs(self.idxs[self.cur_idx])

        batch_records = self.list_buffer_samples[didx][tidx]

        with open(self.data_info_fps[didx]) as f:
            data_info = json.load(f)
        ball_csvs = [pd.read_csv(fp, index_col=0) for fp in self.ball_csv_fpss[didx]]

        # input data
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
            batch_images[0, i][:, :img.shape[1], :img.shape[2]] = img

        curr_step = extract_integer(batch_records[-1])
        for j in range(self.num_balls[didx]):
            # label info
            record_label = [data_info[str(j)]['label']]
            batch_labels[0, j] = torch.FloatTensor(record_label)

            # object proposal info
            record_prop_state = list(ball_csvs[j].iloc[curr_step][['pose_x', 'pose_y']]) + [data_info[str(j)]['radius']]
            batch_prop_states[0, j, 0] = scale2scale(record_prop_state[0], -1.0, 1.0, 0.0, 256.0)
            batch_prop_states[0, j, 1] = scale2scale(record_prop_state[1], -1.0, 1.0, 256.0, 0.0)
            batch_prop_states[0, j, 2] = scale2scale(record_prop_state[2], 0.0, 1.0, 0.0, 256.0 // 2)

            # future steps info
            for i, future_step in enumerate(self.cfg.VAL.prediction_timesteps):
                record_state = ball_csvs[j].iloc[curr_step + future_step][['pose_x', 'pose_y', 'vel_lin_x', 'vel_lin_y']]

                batch_ext_states[0, i, j, 0] = scale2scale(record_state[0], -1.0, 1.0, 0.0, 256.0)
                batch_ext_states[0, i, j, 1] = scale2scale(record_state[1], -1.0, 1.0, 256.0, 0.0)
                batch_ext_states[0, i, j, 2] = record_state[2] * 256.0 / (self.cfg.DATASET.fps * 2)
                batch_ext_states[0, i, j, 3] = - record_state[3] * 256.0 / (self.cfg.DATASET.fps * 2)

        output = dict()
        output['img_data'] = batch_images
        output['state_label'] = batch_labels
        output['state_prop'] = batch_prop_states
        output['future_states'] = batch_ext_states
        return output

    def __len__(self):
        return sum([max(0, num_sample - max(self.cfg.VAL.prediction_timesteps)) for num_sample in self.num_samples])


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, **kwargs):
        super(TestDataset, self).__init__(**kwargs)
        self.cur_idx = -1
        self.img_size = cfg.DATASET.img_size
        self.buffer_size = cfg.DATASET.buffer_size

        self.data_fp = cfg.TEST.list_test
        self.list_buffer_samples = None
        self.num_samples = -1

        # parse the image files
        self.parse_input_list(**kwargs)
        print(f'# buffer samples: {self.num_samples - self.buffer_size}')

    def parse_input_list(self, max_sample=-1, start_idx=-1, end_idx=-1):
        if not os.path.exists(self.data_fp):
            raise NotADirectoryError
        list_sample = []
        fp_filter = re.compile(fr'{TEST_BUFFER_PREFIX}_\d+_{TEST_TIMESTEP_PREFIX}_\d+.png')
        img_fps = [fp for fp in os.listdir(self.data_fp) if re.match(fp_filter, fp)]
        for fp in sorted(img_fps, key=extract_integers):
            list_sample.append(fp)

        if max_sample > 0:
            list_sample = list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:  # divide file list
            list_sample = list_sample[start_idx:end_idx]

        num_sample = len(list_sample) // self.buffer_size
        assert num_sample > 0
        self.num_samples = num_sample
        self.list_buffer_samples = [list_sample[i * self.buffer_size:(i + 1) * self.buffer_size] for i in
                                    range(num_sample)]
        assert self.check_buffer_list(self.list_buffer_samples)

    def check_buffer_list(self, input_list):
        for buffer_idx, buffer in enumerate(input_list):
            # check buffer corresponds to model
            if len(buffer) != self.buffer_size:
                return False

            previous_timestep = -1
            for timestep_file in buffer:
                file_parts = timestep_file.split('_')
                current_buffer = int(file_parts[1])
                current_timestep = int(file_parts[3].split('.')[0])

                # check buffer ordering is correct
                if current_buffer != buffer_idx or current_timestep != previous_timestep + 1:
                    return False
                previous_timestep = current_timestep
        return True

    def __getitem__(self, index):
        batch_images = torch.zeros(1, self.buffer_size, 3, self.img_size[0], self.img_size[1])

        self.cur_idx += 1
        if self.cur_idx >= self.num_samples:
            self.cur_idx = 0

        batch_records = self.list_buffer_samples[self.cur_idx]

        for i in range(self.buffer_size):
            record_img_fn = batch_records[i]

            # load image and label
            image_path = os.path.join(self.data_fp, record_img_fn)
            img = Image.open(image_path).convert('RGB')
            if not ((img.width == self.img_size[0]) and (img.height == self.img_size[1])):
                img = img_resize(img, (self.img_size[0], self.img_size[1]), interp='bilinear')
            # image transform, to torch float tensor 3xHxW
            img = img_transform(img)
            # put into batch arrays
            batch_images[0, i][:, :img.shape[1], :img.shape[2]] = img

        output = dict()
        output['img_data'] = batch_images
        return output

    def __len__(self):
        return self.num_samples
