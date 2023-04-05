# -*- coding: utf-8 -*-
import pybullet as pb
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import os
import time
import json
import argparse


# project directory
PROJECT_DIR = '/home/jonathan/Documents/diss/intuitive_physics/data'

# frame rate parameters PyBullet
PB_FPS = 240
# seconds per frame PyBullet
PB_SEC_PER_FRAME = 1 / PB_FPS

# camera view matrix
CAM_VIEW_MATRIX = pb.computeViewMatrix(
    cameraEyePosition=[0, 0, 2],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])

# camera projection matrix
# perfect fov (not really) 53.13010235415598
CAM_PROJECTION_MATRIX = pb.computeProjectionMatrixFOV(
    fov=55.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=3.1)

DEFAULT_LIGHT_DIRECTION = [0, 0, 4]

# spawning balls slightly above floor
SURFACE_SPHERE_OFFSET = 0.01
# dfa
DEFAULT_RESTITUTION = 0.9

# minimum x in metres for urdf bounds
WORLD_X_MIN = -0.5
# maximum x in metres for urdf bounds
WORLD_X_MAX = 0.5
# minimum y in metres for urdf bounds
WORLD_Y_MIN = -0.5
# maximum y in metres for urdf bounds
WORLD_Y_MAX = 0.5

# ball radius values in metres
RAD_VALUES = np.array([0.03, 0.06, 0.09])
# ball mass values in kilograms
MASS_VALUES = np.array([0.2, 0.4, 0.6])

# rolling friction coefficient values
#FRIC_VALUES = np.array([10 ** (-4), 5 * 10 ** (-4), 10 ** (-3)])
FRIC_VALUES = np.array([10 ** (-4), 10 ** (-3)])
# linear velocity range in ms⁻1
MAX_ABS_VEL = 2.5
# variety of colours for balls
BALL_COLOR_OPT = np.array([
    [186, 76, 64, 255], [69, 192, 151, 255], [87, 52, 133, 255],
    [168, 174, 62, 255], [136, 116, 217, 255], [105, 160, 80, 255],
    [190, 100, 178, 255], [188, 125, 54, 255], [93, 138, 212, 255],
    [185, 73, 115, 255]]) / 255.0

# ball csv properties
BALL_SIM_COL_NAMES = ['timestep', 'pose_x', 'pose_y', 'pose_z',
                      'pose_rx', 'pose_ry', 'pose_rz', 'pose_rw',
                      'vel_lin_x', 'vel_lin_y', 'vel_lin_z',
                      'vel_ang_x', 'vel_ang_y', 'vel_ang_z']


def scale2scale(value, oldMin=-1.0, oldMax=1.0, newMin=-1.0, newMax=1.0):
    """
    Convert linear scale (min/max) to another linear scale (min/max)
    value: value to be converted
    oldMin: old minimum value
    oldMax: old maximum value
    newMin: new minimum value
    newMax: new maximum value
    return: value mapped from old range to new range
    """
    oSpan = oldMax - oldMin
    nSpan = newMax - newMin
    result = ((value - oldMin) / oSpan) * nSpan + newMin
    return result


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Synthetic Pool Ball Collision Data Generator",
        description="Simulate top-down synthetic pool ball collision videos")

    parser.add_argument("--data-folder", default='data',
                        help="folder where all data is saved (default: 'data')")

    parser.add_argument("--sim-name", default='sample',
                        help="simulation folder prefix in data folder (default: 'sample')")

    parser.add_argument("--img-prefix", default='timestep_',
                        help="prefix for simulation image files (default: 'timestep_')")

    parser.add_argument("--img-type", default='.png',
                        help="image type for simulation files (default: '.png')")

    parser.add_argument("--csv-prefix", default='ball_',
                        help="simulation balls csv data prefix (default: 'ball_')")

    parser.add_argument("--json-prefix", default='info',
                        help="simulation balls csv data prefix (default: 'info')")

    parser.add_argument("--use-gui", action='store_true', default=False,
                        help="visualize with PyBullet GUI (default: False)")

    parser.add_argument("--use-render", type=bool, default=True,
                        help="render simulation images (default: True)")

    parser.add_argument("--num-sims", type=int, default=1,
                        help="number of simulations to run (default: 1)")

    parser.add_argument("--fps", type=int, default=30,
                        help="desired frames per second. Only applies if render True (default: 30)")

    parser.add_argument("--duration", type=float, default=2.0,
                        help="simulation duration in seconds (default: 2.0)")

    parser.add_argument("--num-balls", type=int, default=3,
                        help="number of balls in simulation (default: 3)")

    parser.add_argument("--same-vis", action='store_true', default=False,
                        help="keep the appearance same for all balls (default: False)")

    parser.add_argument("--same-phys", action='store_true', default=False,
                        help="keep the physics same for all balls (default: False)")

    parser.add_argument("--rand-seed", type=int, default=42,
                        help="random seed for numpy (default=42)")

    parser.add_argument("--mode", default='none',
                        help="mode for generation specific experiment data (default: 'none')")

    return parser


def initialize_simulator(use_gui):
    """Initialize PyBullet"""
    if use_gui:
        sim_mode = pb.GUI
    else:
        sim_mode = pb.DIRECT
    physicsClient = pb.connect(sim_mode)


def initialize_environment(data_folder):
    """Create environment"""
    urdf_filepath = os.path.join(PROJECT_DIR, 'urdf')
    plane_filepath = os.path.join(urdf_filepath, 'plane', 'plane.urdf')
    wall1_filepath = os.path.join(urdf_filepath, 'wall1', 'wall1.urdf')
    wall2_filepath = os.path.join(urdf_filepath, 'wall2', 'wall2.urdf')
    wall3_filepath = os.path.join(urdf_filepath, 'wall3', 'wall3.urdf')
    wall4_filepath = os.path.join(urdf_filepath, 'wall4', 'wall4.urdf')

    planeId = pb.loadURDF(plane_filepath)
    wall1Id = pb.loadURDF(wall1_filepath)
    wall2Id = pb.loadURDF(wall2_filepath)
    wall3Id = pb.loadURDF(wall3_filepath)
    wall4Id = pb.loadURDF(wall4_filepath)

    # set environment conditions
    pb.setGravity(0, 0, -9.81)

    # environment object id dictionary
    env_id_dict = {'planeId': planeId, 'wallIds': [wall1Id, wall2Id, wall3Id, wall4Id]}
    return env_id_dict


def ballProps2Label(idxs):
    """
    Converts indexes for ball properties to unique index.
    """
    shape = [len(MASS_VALUES), len(FRIC_VALUES)]
    idx = 0
    offset = 1
    for i in range(len(shape) - 1, -1, -1):
        idx += idxs[i] * offset
        offset *= shape[i]
    return idx


def label2BallProps(idx):
    """
    Converts unique index for intrinsic properties to indexes for each intrinsic property
    indices of an element in a flattened numpy array.
    """
    shape = [len(MASS_VALUES), len(FRIC_VALUES)]
    num_dims = len(shape)
    offset = 1
    idxs = [0] * num_dims
    for i in range(num_dims - 1, -1, -1):
        idxs[i] = idx // offset % shape[i]
        offset *= shape[i]
    return idxs


def initialize_balls(num_balls, same_phys, same_vis, mode):
    """Create balls"""

    # randomized ball properties
    if mode == 'none':
        if same_phys:
            ball_mass_idx = np.array([np.random.randint(len(MASS_VALUES))] * num_balls, dtype=int)
            ball_fric_idx = np.array([np.random.randint(len(FRIC_VALUES))] * num_balls, dtype=int)
        else:
            ball_mass_idx = np.random.randint(len(MASS_VALUES), size=num_balls, dtype=int)
            ball_fric_idx = np.random.randint(len(FRIC_VALUES), size=num_balls, dtype=int)
        if same_vis:
            ball_radi_idx = np.array([np.random.randint(len(RAD_VALUES))] * num_balls, dtype=int)
            ball_color_idx = np.array([np.random.randint(len(BALL_COLOR_OPT))] * num_balls, dtype=int)
        else:
            ball_radi_idx = np.random.randint(len(RAD_VALUES), size=num_balls, dtype=int)
            ball_color_idx = np.random.randint(len(BALL_COLOR_OPT), size=num_balls, dtype=int)
        labeler = ballProps2Label
    elif mode == 'predict_friction':
        ball_mass_idx = np.zeros(num_balls, dtype=int)
        ball_fric_idx = np.random.randint(len(FRIC_VALUES), size=num_balls, dtype=int)
        ball_radi_idx = np.ones(num_balls, dtype=int)
        ball_color_idx = np.zeros(num_balls, dtype=int)
        labeler = lambda x: x[1]
    elif mode == 'predict_mass':
        ball_mass_idx = ball_radi_idx = np.random.randint(min(len(MASS_VALUES), len(RAD_VALUES)), size=num_balls, dtype=int)
        ball_fric_idx = np.zeros(num_balls, dtype=int)
        ball_color_idx = np.zeros(num_balls, dtype=int)
        labeler = lambda x: x[0]
    elif mode == 'predict_friction_mass_dependent':
        ball_mass_idx = ball_radi_idx = np.random.randint(min(len(MASS_VALUES), len(RAD_VALUES)), size=num_balls, dtype=int)
        ball_fric_idx = ball_color_idx = np.random.randint(min(len(FRIC_VALUES), len(BALL_COLOR_OPT)), size=num_balls, dtype=int)
        labeler = ballProps2Label
    elif mode == 'predict_friction_mass_independent':
        ball_mass_idx = ball_radi_idx = np.random.randint(min(len(MASS_VALUES), len(RAD_VALUES)), size=num_balls, dtype=int)
        ball_fric_idx = np.random.randint(len(FRIC_VALUES), size=num_balls, dtype=int)
        ball_color_idx = np.zeros(num_balls, dtype=int)
        labeler = ballProps2Label
    else:
        raise NotImplementedError(f'mode not implemented: {mode}')

    ballMass = MASS_VALUES[ball_mass_idx]
    ballFriction = FRIC_VALUES[ball_fric_idx]
    ballRadi = RAD_VALUES[ball_radi_idx]
    ballColor = BALL_COLOR_OPT[ball_color_idx]
    ballInitLinVel = [scale2scale(np.append(np.random.rand(2), 0)) for _ in
                      range(num_balls)]
    ballInitAngVel = [[0, 0, 0]] * num_balls

    # create balls in simulation
    ballIds = []
    ballPropIdxs = []
    ballInitPos = []
    for i in range(num_balls):
        # create new position that does not overlap with walls or balls
        allSafePos = False
        while not allSafePos:
            # new position guaranteed not to overlap with walls
            initPos = np.array(
                [scale2scale(np.random.rand()),
                 scale2scale(np.random.rand()),
                 ballRadi[i] + SURFACE_SPHERE_OFFSET])
            # check if overlaps with other balls
            isSafePos = True
            for j in range(i):
                isSafePos = (ballRadi[i] + ballRadi[j]) < np.sum(np.abs(ballInitPos[j] - initPos))
                if not isSafePos:
                    break
            if isSafePos:
                allSafePos = True
        ballInitPos.append(initPos)

        # adding balls to simulation
        colBallId = pb.createCollisionShape(pb.GEOM_SPHERE, radius=ballRadi[i])
        visualBallId = pb.createVisualShape(pb.GEOM_SPHERE, radius=ballRadi[i], rgbaColor=ballColor[i])
        ballId = pb.createMultiBody(baseMass=ballMass[i],
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=colBallId,
                                    baseVisualShapeIndex=visualBallId,
                                    basePosition=ballInitPos[i])
        # set object properties
        pb.changeDynamics(ballId, -1, rollingFriction=ballFriction[i], contactProcessingThreshold=0, restitution=DEFAULT_RESTITUTION,
                          linearDamping=0, angularDamping=0)
        pb.resetBaseVelocity(ballId, ballInitLinVel[i], ballInitAngVel[i])

        ballPropIdx = int(labeler([ball_mass_idx[i], ball_fric_idx[i]]))
        ballPropIdxs.append(ballPropIdx)

        # keeping object Id
        ballIds.append(ballId)

    return ballIds, ballPropIdxs, ballColor, ballRadi, ballMass, ballFriction


def initialize_directory(data_folder, sim_name):
    """Create simulation directory"""
    video_fp = os.path.join(PROJECT_DIR, data_folder, sim_name + '_0')
    i = 1
    while os.path.exists(video_fp):
        video_fp = os.path.join(PROJECT_DIR, data_folder, sim_name) + f'_{i}'
        i += 1
    try:
        os.makedirs(video_fp)
    except OSError as error:
        print(f"Directory '{video_fp}' cannot be created")
        exit()
    return video_fp


def save_simulation_info(video_fp, json_prefix, des_fps, num_balls, ballPropxIdxs, ballColors, ballRadi, ballMass,
                         ballFriction):
    # saving object properties
    objects_info_filepath = os.path.join(video_fp, json_prefix + '.json')
    objects_info = {'fps': des_fps, 'num_balls': num_balls}
    for i in range(num_balls):
        objects_info[i] = {
            'label': ballPropxIdxs[i],
            'color': list(ballColors[i]),
            'radius': ballRadi[i],
            'mass': ballMass[i],
            'friction': ballFriction[i]
        }
    with open(objects_info_filepath, "w") as f:
        json.dump(objects_info, f)


def save_ball_infos(video_fp, csv_prefix, num_balls, ball_sim_values):
    # save pose and velocity ground truth
    for i in range(num_balls):
        df = pd.DataFrame(ball_sim_values[i, :, :].T, columns=BALL_SIM_COL_NAMES)
        csv_name = f"{csv_prefix}{i}.csv"
        ball_csv_filepath = os.path.join(video_fp, csv_name)
        df.to_csv(ball_csv_filepath, index=False)


def run_simulation(data_folder, sim_name, img_prefix, img_type, csv_prefix, json_prefix, use_gui, use_render, des_fps,
                   duration, num_balls, same_phys, same_vis, mode):
    initialize_simulator(use_gui)

    env_id_dict = initialize_environment(data_folder)
    ballIds, ballPropIdxs, ballColors, ballRadi, ballMass, ballFriction = initialize_balls(num_balls, same_phys, same_vis, mode)

    video_fp = initialize_directory(data_folder, sim_name)

    save_simulation_info(video_fp, json_prefix, des_fps, num_balls, ballPropIdxs, ballColors, ballRadi, ballMass,
                         ballFriction)

    # desired total number of timesteps
    pb_total_timesteps = int(duration * PB_FPS)
    # desired total number of timesteps
    desired_total_timesteps = int(duration * des_fps)
    # set up ball pose and velocity info array
    ball_sim_values = -1 * np.ones((num_balls, len(BALL_SIM_COL_NAMES), desired_total_timesteps))

    # run simulation
    if use_gui:
        frame_init_time = time.time()
    # -1 to create to render a frame at first timestep
    prev_num_des_frames = -1
    for timestep in range(pb_total_timesteps):
        # render a video frame
        curr_num_des_frames = int(timestep * des_fps / PB_FPS)
        if curr_num_des_frames > prev_num_des_frames:
            prev_num_des_frames = curr_num_des_frames

            # storing physical state at given timestep
            for i in range(num_balls):
                ballId = ballIds[i]
                ball_sim_values[i, :, curr_num_des_frames] = np.concatenate([[curr_num_des_frames],
                                                                             np.concatenate(
                                                                                 pb.getBasePositionAndOrientation(
                                                                                     ballIds[i])),
                                                                             np.concatenate(
                                                                                 pb.getBaseVelocity(ballIds[i]))
                                                                             ])

            # taking image
            if use_render:
                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                    width=256,
                    height=256,
                    viewMatrix=CAM_VIEW_MATRIX,
                    projectionMatrix=CAM_PROJECTION_MATRIX,
                    lightDirection=DEFAULT_LIGHT_DIRECTION)
                # save image
                im = Image.fromarray(rgbImg)
                img_name = f"{img_prefix}{curr_num_des_frames}{img_type}"
                img_filepath = os.path.join(video_fp, img_name)
                im.save(img_filepath)

        pb.stepSimulation()

        if use_gui:
            frame_elapsed_sec = time.time() - frame_init_time
            if frame_elapsed_sec < PB_SEC_PER_FRAME:
                time.sleep(PB_SEC_PER_FRAME - frame_elapsed_sec)
            frame_init_time = time.time()

    save_ball_infos(video_fp, csv_prefix, num_balls, ball_sim_values)

    # end simulation
    pb.disconnect()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    data_folder = args.data_folder
    sim_name = args.sim_name
    img_prefix = args.img_prefix
    img_type = args.img_type
    csv_prefix = args.csv_prefix
    json_prefix = args.json_prefix

    use_gui = args.use_gui
    use_render = args.use_render

    num_sims = args.num_sims
    des_fps = args.fps
    duration = args.duration
    num_balls = args.num_balls
    same_phys = args.same_phys
    same_vis = args.same_vis
    mode = args.mode

    np.random.seed(args.rand_seed)

    for sim_iter in tqdm(range(num_sims)):
        run_simulation(data_folder, sim_name,
                       img_prefix, img_type,
                       csv_prefix, json_prefix,
                       use_gui, use_render,
                       des_fps, duration, num_balls, same_phys, same_vis, mode)
