# -*- coding: utf-8 -*-
# This file is code altered from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# System libs
import os
import argparse
# Numerical libs
import torch
from tqdm import tqdm
from PIL import Image
# Our libs
from utils.metric import setup_logger
from utils.config import cfg
from utils.dataset import TestDataset
from utils.pose import tensor_to_cv2
from utils.models.lib import user_scattered_collate, async_copy_to
from utils.models.models import ModelBuilder, PerceptualModule
from utils.physics.simulators import PhysicsEngineBuilder
from utils.graphics.renderers import GraphicsEngineBuilder
from data.data_gen import label2BallProps, MASS_VALUES, FRIC_VALUES


def save_output(img, buffer, step, cfg):
    img = tensor_to_cv2(img)
    output_fp = os.path.join(cfg.TEST.result, f'buffer_{buffer}_timestep_{step}.png')
    Image.fromarray(img).save(output_fp)


def test(perceptual_module, physics_engine, graphics_engine, loader, gpu, cfg):
    perceptual_module.eval()

    max_steps = max(cfg.TEST.prediction_timesteps)

    pbar = tqdm(total=len(loader))
    for buffer_idx, batch_data in enumerate(loader):
        # process data
        batch_data = batch_data[0]

        with torch.no_grad():
            feed_dict = batch_data.copy()
            feed_dict = async_copy_to(feed_dict, gpu)
            predicted_params, intrinsic_params, extrinsic_params = perceptual_module(feed_dict)

        num_balls = len(predicted_params)
        for j in range(num_balls):
            predicted_param_idxs = label2BallProps(predicted_params[j])
            intrinsic_params[j, -2] = MASS_VALUES[predicted_param_idxs[0]]
            intrinsic_params[j, -1] = FRIC_VALUES[predicted_param_idxs[1]]

        physics_engine.start(intrinsic_params, extrinsic_params)
        for step in range(max_steps):
            #print(step, physics_engine.get_extrinsic_parameters())
            physics_engine.step()
            if (step+1) in cfg.TEST.prediction_timesteps:
                img = graphics_engine.get_img()
                save_output(img, buffer_idx, step + 1, cfg)
        physics_engine.stop()

        pbar.update(1)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    optical_flow = ModelBuilder.build_optical_flow(cfg.MODEL.optical_flow)
    input_dim = cfg.MODEL.include_images * cfg.DATASET.buffer_size * 3 \
                + cfg.MODEL.include_optical_flow * (cfg.DATASET.buffer_size - 1) * 2
    net_perceptual = ModelBuilder.build_perceptual(arch='resnet18', input_dim=input_dim,
                                                   num_classes=cfg.DATASET.num_classes,
                                                   weights=cfg.MODEL.weights_perceptual)
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    perceptual_module = PerceptualModule(optical_flow, net_perceptual, crit,
                                         buffer_size=cfg.DATASET.buffer_size,
                                         include_images=cfg.MODEL.include_images,
                                         include_optical_flow=cfg.MODEL.include_optical_flow)

    physics_engine = PhysicsEngineBuilder.build_physics_engine(cfg.TEST.physics_engine, cfg)

    graphics_engine = GraphicsEngineBuilder.build_graphics_engine(cfg.TEST.graphics_engine, cfg, physics_engine)

    # Dataset and Loader
    dataset_test = TestDataset(cfg)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        drop_last=True)

    perceptual_module.cuda()

    # Main loop
    test(perceptual_module, physics_engine, graphics_engine, loader_test, gpu, cfg)

    print('Inference done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Visual De-animation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image path, or a directory name"
    )
    parser.add_argument(
        "--cfg",
        default="config/predict_friction-farneback-resnet18-pybullet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
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
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.perceptual = cfg.MODEL.perceptual.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_perceptual = os.path.join(
        cfg.DIR, 'perceptual_' + cfg.TEST.checkpoint)

    # generate testing image list
    cfg.TEST.list_test = args.imgs

    assert os.path.exists(cfg.MODEL.weights_perceptual), "checkpoint does not exist!"

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
