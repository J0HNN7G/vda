# -*- coding: utf-8 -*-
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='intuitive_physics',
    version='1.0.0',
    author='Jonathan Gustafsson Frennert',
    description='Data generation, training, testing and validation for non-prehensile manipulation of pool balls using visual de-animation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/J0HNN7G/intuitive_physics',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD 3 License',
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'numpy',
        'pybullet',
        'torch',
        'torchvision',
        'Pillow',
        'opencv-python'
        'pandas',
        'matplotlib',
        'tqdm',
        'yacs'
    ]
)
