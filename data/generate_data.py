import pybullet as pb
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import os
import time
import csv
import json
import argparse


# project directory
PROJECT_DIR = '/home/jonathan/Documents/diss/intuitive_physics'

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
CAM_PROJECTION_MATRIX = pb.computeProjectionMatrixFOV(
        fov=55.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=3.1)

# minimum x in metres for world bounds
WORLD_X_MIN = -0.5
# maximum x in metres for world bounds
WORLD_X_MAX = 0.5
# minimum y in metres for world bounds
WORLD_Y_MIN = -0.5
# maximum y in metres for world bounds
WORLD_Y_MAX = 0.5

# ball radius range in metres
INC_RAD = 0.05 # min and increment
MAX_RAD = 0.1
# ball mass range in kilograms
INC_MASS = 0.1 # min and increment
MAX_MASS = 1
# rolling friction range (10 ** (FRIC_EXP) )
MIN_FRIC_EXP = -4
MAX_FRIC_EXP = -2
# linear velocity range in ms⁻1
MAX_ABS_VEL = 2.5
# variety of colours for balls
BALL_COLOR_OPT = np.array([
    [186, 76, 64, 255], [69, 192, 151, 255], [87, 52, 133, 255],
    [168, 174, 62, 255], [136, 116, 217, 255], [105, 160, 80, 255],
    [190, 100, 178, 255], [188, 125, 54, 255], [93, 138, 212, 255],
    [185, 73, 115, 255]]) / 255.0
# ball csv properties
BALL_SIM_COL_NAMES = ['timestep', 'pose_x',  'pose_y', 'pose_z',
                      'pose_rx', 'pose_ry','pose_rz', 'pose_rw',
                      'vel_lin_x', 'vel_lin_y', 'vel_lin_z',
                      'vel_ang_x', 'vel_ang_y', 'vel_ang_z']


def get_parser():
    parser = argparse.ArgumentParser(
                prog="Synthetic Pool Ball Collision Data Generator",
                description="Simulate top-down synthetic pool ball collision videos")

    parser.add_argument("--data-folder", default='data',
                        help = "folder where all data is saved")

    parser.add_argument("--sim-name", default='sample',
                        help = "simulation folder prefix in data folder (default: 'sample')")

    parser.add_argument("--img-prefix", default='timestep_',
                        help = "prefix for simulation image files (default: 'timestep_')")

    parser.add_argument("--img-type", default='.png',
                        help = "image type for simulation files (default: '.png')")

    parser.add_argument("--csv-prefix", default='ball_',
                        help = "simulation balls csv data prefix (default: 'ball_')")

    parser.add_argument("--json-prefix", default='info',
                        help = "simulation balls csv data prefix (default: 'info')")

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

    parser.add_argument("--rand-seed", type=int, default=42,
                        help = "random seed for numpy (default=42)")

    return parser


def newScale(oldValue, oldMin=0, oldMax=1, newMin=-1, newMax=1):
    """Change value to different linear scale"""
    oldRange = (oldMax - oldMin)
    newRange = (newMax - newMin)
    newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin
    return newValue


def initialize_simulator(use_gui):
    """Initialize PyBullet"""
    if use_gui:
        sim_mode = pb.GUI
    else:
        sim_mode = pb.DIRECT
    physicsClient = pb.connect(sim_mode)


def initialize_environment():
    """Create environment"""
    world_filepath = os.path.join(PROJECT_DIR, 'world')
    plane_filepath = os.path.join(world_filepath, 'plane', 'plane.urdf')
    wall1_filepath = os.path.join(world_filepath, 'wall1', 'wall1.urdf')
    wall2_filepath = os.path.join(world_filepath, 'wall2', 'wall2.urdf')
    wall3_filepath = os.path.join(world_filepath, 'wall3', 'wall3.urdf')
    wall4_filepath = os.path.join(world_filepath, 'wall4', 'wall4.urdf')

    planeId = pb.loadURDF(plane_filepath)
    wall1Id = pb.loadURDF(wall1_filepath)
    wall2Id = pb.loadURDF(wall2_filepath)
    wall3Id = pb.loadURDF(wall3_filepath)
    wall4Id = pb.loadURDF(wall4_filepath)

    # set environment conditions
    pb.setGravity(0,0,-9.81)

    # environment object id dictionary
    env_id_dict = { 'planeId' : planeId, 'wallIds' : [wall1Id, wall2Id, wall3Id, wall4Id]}
    return env_id_dict


def initialize_balls(num_balls):
    """Create balls"""

    # randomized ball properties
    ballColors = BALL_COLOR_OPT[np.random.choice(len(BALL_COLOR_OPT), num_balls)]
    ballRadi = INC_RAD + np.random.randint(int(MAX_RAD / INC_RAD), size=num_balls) * INC_RAD
    ballMass = INC_MASS + np.random.randint(int(MAX_MASS / INC_MASS), size=num_balls) * INC_MASS
    ballFriction = np.power(10.0, np.random.randint(low=MIN_FRIC_EXP, high=MAX_FRIC_EXP, size=num_balls))
    ballInitLinVel = [ newScale(np.append(np.random.rand(2), 0), newMin=-MAX_ABS_VEL, newMax=MAX_ABS_VEL) for _ in range(num_balls)]
    ballInitAngVel = [[0,0,0]] * num_balls

    # create balls in simulation
    ballIds = []
    ballInitPos = []
    for i in range(num_balls):
        # create new position that does not overlap with walls or balls
        allSafePos = False
        while not allSafePos:
            # new position guaranteed not to overlap with walls
            initPos = np.array([newScale(np.random.rand(), newMin=WORLD_X_MIN+ballRadi[i], newMax=WORLD_X_MAX-ballRadi[i]),
                                newScale(np.random.rand(), newMin=WORLD_Y_MIN+ballRadi[i], newMax=WORLD_Y_MAX-ballRadi[i]),
                                ballRadi[i]+0.01])
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
        visualBallId = pb.createVisualShape(pb.GEOM_SPHERE, radius=ballRadi[i], rgbaColor=ballColors[i])
        ballId = pb.createMultiBody(baseMass=ballMass[i],
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=colBallId,
                                    baseVisualShapeIndex=visualBallId,
                                    basePosition=ballInitPos[i])
        # set object properties
        pb.changeDynamics(ballId, -1, rollingFriction=ballFriction[i], contactProcessingThreshold=0, restitution=0.9, linearDamping=0, angularDamping=0)
        pb.resetBaseVelocity(ballId, ballInitLinVel[i], ballInitAngVel[i])

        # keeping object Id
        ballIds.append(ballId)

    return ballIds, ballColors, ballRadi, ballMass, ballFriction


def initialize_directory(data_folder, sim_name):
    """Create simulation directory"""
    video_fp = os.path.join(PROJECT_DIR, data_folder, sim_name + '_0')
    i = 1
    while os.path.exists(video_fp):
        video_fp = os.path.join(PROJECT_DIR, data_folder, sim_name) + f'_{i}'
        i += 1
    try:
        os.mkdir(video_fp)
    except OSError as error:
        print(f"Directory '{video_fp}' cannot be created")
        exit()
    return video_fp


def save_simulation_info(video_fp, json_prefix, ballColors, ballRadi, ballMass, ballFriction):
    # saving object properties
    objects_info_filepath = os.path.join(video_fp, json_prefix + '.json')
    objects_info = {}
    for i in range(num_balls):
        objects_info[i]  = {
        'color' : list(ballColors[i]),
        'radius' : ballRadi[i],
        'mass' : ballMass[i],
        'friction' : ballFriction[i]
        }
    with open(objects_info_filepath, "w") as f:
        json.dump(objects_info, f)


def save_ball_infos(video_fp, csv_prefix, num_balls, ball_sim_values):
    # save pose and velocity ground truth
    for i in range(num_balls):
        df = pd.DataFrame(ball_sim_values[i, :, :].T, columns=BALL_SIM_COL_NAMES)
        csv_name = f"{csv_prefix}{i}.csv"
        ball_csv_filepath = os.path.join(video_fp, csv_name)
        df.to_csv(ball_csv_filepath)


def run_simulation(data_folder, sim_name, img_prefix, img_type, csv_prefix, json_prefix, use_gui, use_render, des_fps, duration, num_balls):
    initialize_simulator(use_gui)

    env_id_dict = initialize_environment()
    ballIds, ballColors, ballRadi, ballMass, ballFriction = initialize_balls(num_balls)
    video_fp = initialize_directory(data_folder, sim_name)

    save_simulation_info(video_fp, json_prefix, ballColors, ballRadi, ballMass, ballFriction)


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
                                                                             np.concatenate(pb.getBasePositionAndOrientation(ballIds[i])),
                                                                             np.concatenate(pb.getBaseVelocity(ballIds[i]))
                                                                             ])

            # taking image
            if use_render:
                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                                                                        width=256,
                                                                        height=256,
                                                                        viewMatrix=CAM_VIEW_MATRIX,
                                                                        projectionMatrix=CAM_PROJECTION_MATRIX,
                                                                        lightDirection=[0, 0, 4])
                # save image
                im = Image.fromarray(rgbImg)
                img_name = f"{img_prefix}{timestep}{img_type}"
                img_filepath = os.path.join(video_fp, img_name)
                im.save(img_filepath)

        pb.stepSimulation()

        if use_gui:
            frame_elapsed_sec = time.time() - frame_init_time
            if frame_elapsed_sec <  PB_SEC_PER_FRAME:
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

    np.random.seed(args.rand_seed)

    for sim_iter in tqdm(range(num_sims)):
        run_simulation(data_folder, sim_name,
                       img_prefix, img_type,
                       csv_prefix, json_prefix,
                       use_gui, use_render,
                       des_fps, duration, num_balls)
