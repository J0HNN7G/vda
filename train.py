# -*- coding: utf-8 -*-
# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from utils.log import setup_logger, AverageMeter
from utils.gpu import parse_devices
from utils.config import cfg
from utils.dataset import TrainMultiDataset
from utils.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from utils.models import ModelBuilder


# train one epoch
def train(net_perceptual, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    net_perceptual.train()

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        net_perceptual.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        loss, acc = net_perceptual(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item() * 100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print(
                f'Epoch: [{epoch}][{i}/{cfg.TRAIN.epoch_iters}], Time: {batch_time.average():.2f}, Data: {data_time.average():.2f}, '
                f'lr_perceptual: {cfg.TRAIN.running_lr_encoder:.6f}, '
                f'Accuracy: {ave_acc.average():.2f}, Loss: {ave_total_loss.average():.6f}')
            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_perceptual, crit) = nets

    dict_perceptual = net_perceptual.state_dict()

    torch.save(
        history,
        f'{cfg.DIR}/history_epoch_{epoch}.pth')
    torch.save(
        dict_perceptual,
        f'{cfg.DIR}/perceptual_epoch_{epoch}.pth')


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizer(nets, cfg):
    (net_perceptual, crit) = nets
    optimizer_perceptual = torch.optim.SGD(
        group_weight(net_perceptual),
        lr=cfg.TRAIN.lr_perceptual,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return optimizer_perceptual


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builder
    net_perceptual = ModelBuilder.build_perceptual(weights=cfg.MODEL.weights_perceptual)

    crit = nn.NLLLoss(ignore_index=-1)

    # Dataset and Loader
    dataset_train = TrainMultiDataset(
        cfg.DATASET.list_train,
        img_size=(256, 256),
        buffer_size=3,
        random_order=True)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print(f'1 Epoch = {cfg.TRAIN.epoch_iters} iters')

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(gpus) > 1:
        net_perceptual = UserScatteredDataParallel(
            net_perceptual,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(net_perceptual)
    net_perceptual.cuda()

    # Set up optimizers
    nets = (net_perceptual, crit)
    optimizer = create_optimizer(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(net_perceptual, iterator_train, optimizer, history, epoch + 1, cfg)

        # checkpointing
        checkpoint(nets, history, cfg, epoch + 1)

    print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Visual De-animation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/same_vis_same_phys-spynet-resnet18-pybullet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputting checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_perceptual = os.path.join(
            cfg.DIR, f'perceptual_epoch_{cfg.TRAIN.start_epoch}.pth')
        assert os.path.exists(cfg.MODEL.weights_perceptual) and \
               os.path.exists(cfg.MODEL.weights_perceptual), "checkpoint does not exist!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_perceptual = cfg.TRAIN.lr_perceptual

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
