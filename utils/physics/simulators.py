# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod

import math


# PyBullet engine
import pybullet as pb
import torch
from data.data_gen import SURFACE_SPHERE_OFFSET, DEFAULT_RESTITUTION, PB_FPS
from utils.mask import scale2scale


class PhysicsEngineBuilder:
    @staticmethod
    def build_physics_engine(arch='pybullet', cfg=None, weights='', **kwargs):
        arch = arch.lower()
        if arch == 'pybullet':
            assert cfg
            physics_engine = PyBulletPhysicsEngine(cfg)
        else:
            raise Exception('Architecture undefined!')
        return physics_engine


class BasePhysicsEngine(ABC):
    """Base class for Physics Engine"""

    @abstractmethod
    def start(self, **kwargs):
        ...

    @abstractmethod
    def stop(self, **kwargs):
        ...

    @abstractmethod
    def step(self, **kwargs):
        ...

    @abstractmethod
    def get_extrinsic_parameters(self, **kwargs):
        ...


class PyBulletPhysicsEngine(BasePhysicsEngine, ABC):
    def __init__(self, cfg):
        self.urdf_folder = cfg.TEST.urdf_folder
        self.desired_fps = cfg.TEST.fps
        self.physicsClientId = None
        self.env_id_dict = None
        self.objIds = None
        self.pb_timestep = -1

    def start(self, intrinsic_params, extrinsic_params, **kwargs):
        assert not self.physicsClientId
        self.physicsClientId = pb.connect(pb.DIRECT)
        self.init_environment()
        self.init_objects(intrinsic_parameters=intrinsic_params, extrinsic_parameters=extrinsic_params)
        self.pb_timestep = 0

    def stop(self, **kwargs):
        assert self.physicsClientId is not None
        pb.disconnect(self.physicsClientId)
        self.physicsClientId = None

    def init_environment(self, **kwargs):
        assert self.physicsClientId is not None
        urdf_filepath = os.path.join(self.urdf_folder, 'urdf')
        plane_filepath = os.path.join(urdf_filepath, 'plane', 'plane.urdf')
        wall1_filepath = os.path.join(urdf_filepath, 'wall1', 'wall1.urdf')
        wall2_filepath = os.path.join(urdf_filepath, 'wall2', 'wall2.urdf')
        wall3_filepath = os.path.join(urdf_filepath, 'wall3', 'wall3.urdf')
        wall4_filepath = os.path.join(urdf_filepath, 'wall4', 'wall4.urdf')

        planeId = pb.loadURDF(plane_filepath, physicsClientId=self.physicsClientId)
        wall1Id = pb.loadURDF(wall1_filepath, physicsClientId=self.physicsClientId)
        wall2Id = pb.loadURDF(wall2_filepath, physicsClientId=self.physicsClientId)
        wall3Id = pb.loadURDF(wall3_filepath, physicsClientId=self.physicsClientId)
        wall4Id = pb.loadURDF(wall4_filepath, physicsClientId=self.physicsClientId)

        # set environment conditions
        pb.setGravity(0, 0, -9.81, physicsClientId=self.physicsClientId)

        # environment object id dictionary
        env_id_dict = {'planeId': planeId, 'wallIds': [wall1Id, wall2Id, wall3Id, wall4Id]}
        self.env_id_dict = env_id_dict

    def init_objects(self, intrinsic_parameters, extrinsic_parameters, **kwargs):
        assert self.physicsClientId is not None
        objIds = []
        num_balls = len(intrinsic_parameters)

        for i in range(num_balls):
            r, g, b, pr, mass, friction = intrinsic_parameters[i]
            px, py, pvx, pvy = extrinsic_parameters[i]

            cx = scale2scale(px, 0.0, 256.0, -1.0, 1.0)
            cy = scale2scale(py, 0.0, 256.0, 1.0, -1.0)  # check if this actually does what it supposed to
            cr = scale2scale(pr, 0.0, 256.0 // 2, 0.0, 1.0)
            
            cvx = pvx / 256.0 * self.desired_fps * 2 # figure out the scaling factor
            cvy = - pvy / 256.0 * self.desired_fps * 2
            
            # adding balls to simulation
            colBallId = pb.createCollisionShape(pb.GEOM_SPHERE, radius=cr)
            visualBallId = pb.createVisualShape(pb.GEOM_SPHERE, radius=cr, rgbaColor=[r, g, b, 1.0])
            ballId = pb.createMultiBody(baseMass=mass,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseCollisionShapeIndex=colBallId,
                                        baseVisualShapeIndex=visualBallId,
                                        basePosition=[cx, cy, cr + SURFACE_SPHERE_OFFSET])
            # set object properties
            pb.changeDynamics(ballId, -1, rollingFriction=friction, contactProcessingThreshold=0,
                              restitution=DEFAULT_RESTITUTION,
                              linearDamping=0, angularDamping=0)
            pb.resetBaseVelocity(ballId, [cvx, cvy, 0], [- cvy / cr, cvx / cr, 0])
            objIds.append(ballId)
        self.objIds = objIds

    def step(self, **kwargs):
        assert self.physicsClientId is not None
        for i in range(int(PB_FPS / self.desired_fps)):
            pb.stepSimulation(physicsClientId=self.physicsClientId)
            self.pb_timestep += 1

    def get_extrinsic_parameters(self, **kwargs):
        assert self.physicsClientId is not None
        extrinsic_parameters = torch.zeros(len(self.objIds), 4)

        for i in range(len(self.objIds)):
            x, y = pb.getBasePositionAndOrientation(self.objIds[i], physicsClientId=self.physicsClientId)[0][:2]
            vx, vy = pb.getBaseVelocity(self.objIds[i])[0][:2]
            extrinsic_parameters[i, ...] = torch.FloatTensor([x, y, vx, vy])

        return extrinsic_parameters
