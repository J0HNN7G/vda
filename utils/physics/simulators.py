# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod

# PyBullet engine
import pybullet as pb
import torch
from data.data_gen import SURFACE_SPHERE_OFFSET, DEFAULT_RESTITUTION, PB_FPS
from utils.mask import scale2scale


class PhysicsEngineBuilder:
    @staticmethod
    def build_physics_engine(arch='pybullet', weights='', **kwargs):
        arch = arch.lower()
        if arch == 'pybullet':
            physics_engine = PyBulletEngine()
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
    def init_environment(self, **kwargs):
        ...

    @abstractmethod
    def init_objects(self, **kwargs):
        ...

    @abstractmethod
    def init_timestep(self, **kwargs):
        ...

    @abstractmethod
    def step(self, **kwargs):
        ...

    @abstractmethod
    def get_extrinsic_parameters(self, **kwargs):
        ...


class PyBulletEngine(BasePhysicsEngine, ABC):
    def __init__(self):
        self.physicsClientId = None
        self.urdf_folder = None
        self.env_id_dict = None
        self.objIds = None
        self.pb_timestep = -1
        self.des_fps = 0

    def start(self, **kwargs):
        assert not self.physicsClientId
        self.physicsClientId = pb.connect(pb.DIRECT)

    def stop(self, **kwargs):
        assert self.physicsClientId
        pb.disconnect(self.physicsClientId)
        self.physicsClientId = None

    def init_environment(self, **kwargs):
        assert self.physicsClientId
        assert self.urdf_folder
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
        assert self.physicsClientId
        objIds = []
        num_balls = len(intrinsic_parameters)
        for i in range(num_balls):
            r, g, b, pr, mass, friction = intrinsic_parameters[i]
            px, py, pvx, pvy = extrinsic_parameters[i]

            cx = scale2scale(px, 0.0, 256.0, -1.0, 1.0)
            cy = scale2scale(py, 0.0, 256.0, 1.0, -1.0)  # check if this actually does what it supposed to
            cr = scale2scale(pr, 0.0, 256.0 // 2, 0.0, 1.0)
            cvx = pvx / 256.0
            cvy = pvy / 256.0

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
            pb.resetBaseVelocity(ballId, [cvx, cvy, 0], [0, 0, 0])
            objIds.append(ballId)
        self.objIds = objIds

    def init_timestep(self, des_fps, **kwargs):
        self.pb_timestep = 0

    def init_fps(self, des_fps):
        self.des_fps = des_fps

    def init_urdf_folder(self, urdf_folder):
        self.urdf_folder = urdf_folder

    def step(self, **kwargs):
        assert self.physicsClientId
        assert (self.des_fps > 0) and (self.pb_timestep > -1)
        while int(self.pb_timestep + 1 * self.des_fps / PB_FPS) == int(self.pb_timestep * self.des_fps / PB_FPS):
            self.pb_timestep += 1
            pb.stepSimulation(physicsClientId=self.physicsClientId)

    def get_extrinsic_parameters(self, **kwargs):
        assert self.physicsClientId
        extrinsic_parameters = torch.zeros(len(self.objIds), 4)

        for i in range(len(self.objIds)):
            x, y = pb.getBasePositionAndOrientation(self.objIds[i], physicsClientId=self.physicsClientId)[0][:1]
            vx, vy = pb.getBaseVelocity(self.objIds[i])[0][:1]
            extrinsic_parameters[i, ...] = torch.FloatTensor([x, y, vx, vy])

        return extrinsic_parameters
