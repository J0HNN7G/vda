# -*- coding: utf-8 -*-
# This file is code altered from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# System libs
import os
import time
import argparse
from tqdm import tqdm
# Numerical libs
import torch
# Our libs
from vda.metric import setup_logger, MetricMeter, label_accuracy, pixel_mse, manhattan_distance, \
    calculate_object_proposal_metrics
from vda.config import cfg
from vda.dataset import ValDataset
from vda.models.lib import user_scattered_collate, async_copy_to
from vda.models.models import ModelBuilder, PerceptualModule
from vda.physics.simulators import PhysicsEngineBuilder
from vda.graphics.renderers import GraphicsEngineBuilder
from data.data_gen import label2BallProps, MASS_VALUES, FRIC_VALUES


def evaluate(perceptual_module, physics_engine, graphics_engine, loader, gpu, cfg):
    prop_rec_meter = MetricMeter()
    prop_prec_meter = MetricMeter()
    prop_f1_meter = MetricMeter()
    prop_acc_meter = MetricMeter()
    prop_iou_meter = MetricMeter()
    label_acc_meter = MetricMeter()
    pixel_mse_meter = MetricMeter()
    position_manhattan_meters = [MetricMeter() for _ in range(len(cfg.VAL.prediction_timesteps))]
    velocity_manhattan_meters = [MetricMeter() for _ in range(len(cfg.VAL.prediction_timesteps))]
    time_meter = MetricMeter()

    perceptual_module.eval()

    max_steps = max(cfg.VAL.prediction_timesteps)

    pbar = tqdm(total=len(loader))
    for buffer_idx, batch_data in enumerate(loader):
        # process data
        batch_data = batch_data[0]

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            feed_dict = batch_data.copy()
            feed_dict = async_copy_to(feed_dict, gpu)
            predicted_params, intrinsic_params, extrinsic_params = perceptual_module(feed_dict)
        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        predicted_params, intrinsic_params, extrinsic_params = predicted_params.cpu(), intrinsic_params.cpu(), extrinsic_params.cpu()

        num_balls = len(predicted_params)
        pred_object_proposal = torch.FloatTensor(num_balls, 3)
        for j in range(num_balls):
            predicted_param_idxs = label2BallProps(predicted_params[j])
            intrinsic_params[j, -2] = MASS_VALUES[predicted_param_idxs[0]]
            intrinsic_params[j, -1] = FRIC_VALUES[predicted_param_idxs[1]]

        # object proposal metrics
        pred_object_proposal = extrinsic_params[:, [0, 1, -1]]
        actual_prop_state = feed_dict['state_prop'][0].cpu()
        prop_acc, prop_recall, prop_precision, prop_f1, prop_iou, true_pos_idxs = calculate_object_proposal_metrics(
            pred_object_proposal, actual_prop_state)
        prop_acc_meter.update(prop_acc)
        prop_rec_meter.update(prop_recall)
        prop_prec_meter.update(prop_precision)
        prop_f1_meter.update(prop_f1)
        prop_iou_meter.update(prop_iou)

        # for rest of calculations
        actual_idxs = [i for i in range(len(true_pos_idxs)) if true_pos_idxs[i] is not None]
        pred_idxs = [true_pos_idxs[i] for i in range(len(true_pos_idxs)) if true_pos_idxs[i] is not None]

        if not actual_idxs:
            pass
        else:
            # label accuracy
            actual_params = feed_dict['state_label'][0][actual_idxs].cpu()
            predicted_params = predicted_params[pred_idxs]
            label_acc = label_accuracy(predicted_params, actual_params).item()
            label_acc_meter.update(label_acc)

            physics_engine.start(intrinsic_params, extrinsic_params)

            # pixel MSE
            pred_img = graphics_engine.get_img()
            actual_img = feed_dict['img_data'][0, -1].cpu()
            img_mse = pixel_mse(pred_img, actual_img)
            pixel_mse_meter.update(img_mse)

            for step in range(1, max_steps + 1):
                physics_engine.step()

                # metrics
                if step in cfg.VAL.prediction_timesteps:
                    pred_extrinsic_params = physics_engine.get_extrinsic_parameters().cpu()
                    pred_extrinsic_params = pred_extrinsic_params[pred_idxs]

                    ave_idx = cfg.VAL.prediction_timesteps.index(step)
                    actual_extrinsic_params = feed_dict['future_states'][0, ave_idx].cpu()
                    actual_extrinsic_params = actual_extrinsic_params[actual_idxs]

                    # position manhattan distance
                    pred_pos = pred_extrinsic_params[:, :2]
                    actual_pos = actual_extrinsic_params[:, :2]
                    pos_manhattan = torch.mean(manhattan_distance(pred_pos, actual_pos)).item()
                    position_manhattan_meters[ave_idx].update(pos_manhattan)

                    # velocity manhattan distance
                    pred_vel = pred_extrinsic_params[:, 2:]
                    actual_vel = actual_extrinsic_params[:, 2:4]  # need to exclude radius

                    vel_manhattan = torch.mean(manhattan_distance(pred_vel, actual_vel)).item()
                    velocity_manhattan_meters[ave_idx].update(vel_manhattan)
            physics_engine.stop()

        pbar.update(1)

    # summary
    print('[Eval Summary]:\n')
    print(f'{"Inference Time:":<15} {time_meter.average():7.4f} ±{time_meter.standard_deviation():7.4f}s \n')

    print('Object Proposals')
    print(f'{"Acc:":<15} {prop_acc_meter.average():7.4f} ±{prop_acc_meter.standard_deviation():7.4f}')
    print(f'{"Recall:":<15} {prop_rec_meter.average():7.4f} ±{prop_rec_meter.standard_deviation():7.4f}')
    print(f'{"Precision:":<15} {prop_prec_meter.average():7.4f} ±{prop_prec_meter.standard_deviation():7.4f}')
    print(f'{"F1:":<15} {prop_f1_meter.average():7.4f} ±{prop_f1_meter.standard_deviation():7.4f}')
    print(f'{"MIoU:":<15} {prop_iou_meter.average():7.4f} ±{prop_iou_meter.standard_deviation():7.4f}\n')

    print('Reconstruction')
    print(f'{"Label Acc:":<15} {label_acc_meter.average():7.4f} ±{label_acc_meter.standard_deviation():7.4f}')
    print(f'{"Pixel MSE:":<15} {pixel_mse_meter.average():7.4f} ±{pixel_mse_meter.standard_deviation():7.4f}\n')

    print(f'Prediction (Manhattan Distance)')
    for i in range(len(cfg.VAL.prediction_timesteps)):
        pos_ave = position_manhattan_meters[i].average()
        vel_ave = velocity_manhattan_meters[i].average()
        pos_std = position_manhattan_meters[i].standard_deviation()
        vel_std = velocity_manhattan_meters[i].standard_deviation()
        print(
            f'Step {cfg.VAL.prediction_timesteps[i]:<3} Mean Pos. Diff: {pos_ave:8.4f} ±{pos_std:8.4f} - Mean Vel. Diff: {vel_ave:8.4f} ±{vel_std:8.4f}')

    results = {
        'inf_time_ave': time_meter.average(),
        'inf_time_std': time_meter.standard_deviation(),
        'prop_acc_ave': prop_acc_meter.average(),
        'prop_acc_std': prop_acc_meter.standard_deviation(),
        'prop_rec_ave': prop_prec_meter.average(),
        'prop_rec_std': prop_prec_meter.standard_deviation(),
        'prop_prec_ave': prop_f1_meter.average(),
        'prop_prec_std': prop_f1_meter.standard_deviation(),
        'prop_f1_ave': prop_iou_meter.average(),
        'prop_f1_std': prop_iou_meter.standard_deviation(),
        'prop_miou_ave': prop_iou_meter.average(),
        'prop_miou_std': prop_iou_meter.standard_deviation(),
        'label_acc_ave': label_acc_meter.average(),
        'label_acc_std': label_acc_meter.standard_deviation(),
        'pixel_mse_ave': pixel_mse_meter.average(),
        'pixel_mse_std': pixel_mse_meter.standard_deviation(),
        'pred_steps': cfg.VAL.prediction_timesteps,
        'pred_pos_ave': [position_manhattan_meters[i].average() for i in
                         range(len(cfg.VAL.prediction_timesteps))],
        'pred_pos_std': [position_manhattan_meters[i].standard_deviation() for i in
                         range(len(cfg.VAL.prediction_timesteps))],
        'pred_vel_ave': [velocity_manhattan_meters[i].average() for i in
                         range(len(cfg.VAL.prediction_timesteps))],
        'pred_vel_std': [velocity_manhattan_meters[i].standard_deviation() for i in
                         range(len(cfg.VAL.prediction_timesteps))],
    }
    return results


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
    dataset_val_fps = [os.path.join(cfg.DATASET.list_val, fp) for fp in os.listdir(cfg.DATASET.list_val)]
    dataset_val = ValDataset(dataset_val_fps, cfg)

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        drop_last=True)

    perceptual_module.cuda()

    # Main loop
    results = evaluate(perceptual_module, physics_engine, graphics_engine, loader_val, gpu, cfg)
    torch.save(
        results,
        f'{cfg.DIR}/eval_{cfg.VAL.checkpoint}')

    print('Evaluation Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Visual De-animation Validation"
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
        help="gpu to use"
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

    # absolute paths of model weights
    cfg.MODEL.weights_perceptual = os.path.join(
        cfg.DIR, 'perceptual_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_perceptual), "checkpoint does not exist!"

    main(cfg, args.gpu)
