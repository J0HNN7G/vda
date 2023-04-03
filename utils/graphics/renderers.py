# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod

# PyBullet engine
import pybullet as pb
from PIL import Image
from data.data_gen import CAM_VIEW_MATRIX, CAM_PROJECTION_MATRIX, DEFAULT_LIGHT_DIRECTION
from utils.dataset import img_transform


class GraphicsEngineBuilder:
    @staticmethod
    def build_graphics_engine(arch='pybullet', cfg=None, physics_engine=None, weights='', **kwargs):
        arch = arch.lower()
        if arch == 'pybullet':
            assert cfg and physics_engine
            graphics_engine = PyBulletGraphicsEngine(cfg, physics_engine)
        else:
            raise Exception('Architecture undefined!')
        return graphics_engine


class BaseGraphicsEngine(ABC):
    """Base class for Graphics Engine"""

    @abstractmethod
    def init_environment(self, **kwargs):
        ...

    @abstractmethod
    def init_objects(self, **kwargs):
        ...

    @abstractmethod
    def set_objects(self, **kwargs):
        ...

    @abstractmethod
    def get_img(self, **kwargs):
        ...


class PyBulletGraphicsEngine(BaseGraphicsEngine, ABC):
    def __init__(self, cfg, physics_engine):
        self.physics_engine = physics_engine
        self.img_size = cfg.DATASET.img_size

    def init_environment(self, **kwargs):
        pass

    def init_objects(self, **kwargs):
        pass

    def set_objects(self, **kwargs):
        pass

    def get_img(self, **kwargs):
        assert self.physics_engine.physicsClientId is not None
        _, _, rgbImg, _, _ = pb.getCameraImage(
            width=self.img_size[0],
            height=self.img_size[1],
            viewMatrix=CAM_VIEW_MATRIX,
            projectionMatrix=CAM_PROJECTION_MATRIX,
            lightDirection=DEFAULT_LIGHT_DIRECTION,
            physicsClientId=self.physics_engine.physicsClientId)
        # save image
        img = Image.fromarray(rgbImg).convert('RGB')
        img = img_transform(img)
        return img
