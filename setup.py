# -*- coding: utf-8 -*-
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='vda',
    version='1.0.0',
    author='Jonathan Gustafsson Frennert',
    description='Unofficial Implementation of Visual De-Animation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/J0HNN7G/vda',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD 3 License',
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'torch',
        'torchvision',  
        'numpy',
        'matplotlib',
        'Pillow',
        'pandas',
        'tqdm',
        'pybullet',
        'opencv-python'
        'wandb',
        'yacs'
    ]
)
