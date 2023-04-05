#!/bin/sh

# do this first
#source $HOME/Documents/diss/intuitive_physics/env/bin/activate

# ablation
#python3 train.py --cfg config/images_only-predict_friction-farneback-resnet18-pybullet.yaml

# ablation
python3 train.py --cfg config/optical_flow_only-predict_friction-farneback-resnet18-pybullet.yaml
python3 train.py --cfg config/optical_flow_only-predict_friction-spynet-resnet18-pybullet.yaml

# experiment 1
#python3 train.py --cfg config/predict_friction-farneback-resnet18-pybullet.yaml
#python3 train.py --cfg config/predict_friction-spynet-resnet18-pybullet.yaml

# experiment 2
#python3 train.py --cfg config/predict_mass-farneback-resnet18-pybullet.yaml
python3 train.py --cfg config/predict_mass-spynet-resnet18-pybullet.yaml

# experiment 3
python3 train.py --cfg config/predict_friction_mass_dependent-farneback-resnet18-pybullet.yaml
python3 train.py --cfg config/predict_friction_mass_dependent-spynet-resnet18-pybullet.yaml

# experiment 4
python3 train.py --cfg config/predict_friction_mass_independent-farneback-resnet18-pybullet.yaml
python3 train.py --cfg config/predict_friction_mass_independent-spynet-resnet18-pybullet.yaml