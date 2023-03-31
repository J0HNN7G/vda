# -*- coding: utf-8 -*-
# This file is code altered from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# System libs
import os
import argparse
# Numerical libs
import torch
from tqdm import tqdm
# Our libs
from utils.metric import setup_logger
from utils.config import cfg
from utils.dataset import TestDataset
from utils.models.lib import user_scattered_collate, async_copy_to
from utils.models.models import ModelBuilder, PerceptualModule
from utils.physics.simulators import PhysicsEngineBuilder
from utils.graphics.renderers import GraphicsEngineBuilder
from data.data_gen import label2BallProps, MASS_VALUES, FRIC_VALUES


def test(perceptual_module, physics_engine, graphics_engine, loader, gpu):
    perceptual_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]

        with torch.no_grad():
            feed_dict = async_copy_to(feed_dict, gpu)
            predicted_params, intrinsic_params, extrinsic_params = perceptual_module(feed_dict)
            predicted_param_idxs = label2BallProps(predicted_params[0])

        physics_engine.start()
        physics_engine.init_environment()
        physics_engine.init_objects(intrinsic_parameters=, extrinsic_parameters=extrinsic_params)
        physics_engine.init_timestep()
        physics_engine.step()
        extrinsic_params = physics_engine.get_extrinsic_parameters()
        img = graphics_engine.get_img()
        physics_engine.stop()

        pbar.update(1)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    optical_flow = ModelBuilder.build_optical_flow()
    input_dim = cfg.MODEL.include_images * cfg.DATASET.buffer_size * 3 \
                + cfg.MODEL.include_optical_flow * (cfg.DATASET.buffer_size - 1) * 2
    net_perceptual = ModelBuilder.build_perceptual(arch='resnet18', input_dim=input_dim, num_classes=cfg.DATASET.num_classes,
                                                   weights=cfg.MODEL.weights_perceptual)
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    perceptual_module = PerceptualModule(optical_flow, net_perceptual, crit,
                                         buffer_size=cfg.DATASET.buffer_size,
                                         include_images=cfg.MODEL.include_images,
                                         include_optical_flow=cfg.MODEL.include_optical_flow)


    # Dataset and Loader
    dataset_test_fps = [os.path.join(cfg.DATASET.list_test, fp) for fp in os.listdir(cfg.DATASET.list_test)]
    dataset_test = TestDataset(
        dataset_test_fps,
        cfg.DATASET)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    perceptual_module.cuda()
    physics_engine = PhysicsEngineBuilder.build_physics_engine(cfg.TEST.physics_engine)
    graphics_engine = GraphicsEngineBuilder.build_graphics_engine(cfg.TEST.graphics_engine)

    if cfg.TEST.physics_engine == 'pybullet':
        physics_engine.init_fps(cfg.TEST.fps)
        physics_engine.init_urdf_folder(cfg.TEST.urdf_folder)

        graphics_engine.set_physics_engine(physics_engine)
        graphics_engine.set_img_size(cfg.DATASET.img_size)


    # Main loop
    test(perceptual_module, physics_engine, graphics_engine, loader_test, gpu)

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

    assert os.path.exists(cfg.MODEL.weights_perceptual), "checkpoint does not exist!"

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
