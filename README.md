# Visual De-Animation in PyTorch

This is an unofficial PyTorch implementation of visual de-animation.

## Framework
The framework is split into optical flow, object representation module, perceptual module and simulation module. Pre-configured models can be found in the ```config``` directory.

Optical Flow:
- Farneback
- SpyNet

Object Represenation Module:
- Concatenated Images
- Encoder-Decoder Bottleneck (TODO)

Perceptual Module:
- AlexNet (TODO)
- GoogleLeNet (TODO)
- Resnet (Extend)
- VisionTransformer (TODO)

Simulation Module:
- PyBullet (Extend)
- gradSim (TODO)

I provide scripts for training on a high-computing cluster in the ```slurm``` directory and scripts for WandB logging in the  ```wandb``` directory. 

## Environment 
The code is developed under the following configurations.

- Hardware: >=1 GPUs for training, >=1 GPU for testing (set [--gpus GPUS] accordingly)
- Software: Ubuntu 20.04.6 LTS, CUDA=11.7, Python=3.8.10
- Dependencies: torch, torchvision, numpy, matplotlib, Pillow, pandas, tqdm, pybullet, opencv-python, wandb, yacs


## Installation
### Pip
```bash
pip install git+https://github.com/J0HNN7G/vda.git
```

### Conda


```
conda create -n vda
conda activate vda

# pytorch (change to desired version)
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# data processing
conda install numpy matplotlib Pillow pandas tqdm

# not on conda
pip install pybullet opencv-python wandb yacs --upgrade-strategy only-if-needed

# vda
git clone git@github.com:J0HNN7G/vda.git
cd vda
python setup.py install
```

Once installed the package can be consumed programmatically
```python
from vda.config import cfg
from vda.models import PerceptualBuilder
```

## Tutorial
We apply the package to system identification on billiards.
### Training
1. Generate the pool ball data. 
```bash
cd data/
chmod +x data_gen.sh
./data_gen.sh
```
2. Train a model by selecting the GPUs (```$GPUS```) and configuration file (```$CFG```) to use. During training, checkpoints by default are saved in folder ```ckpt```.
```bash
python3 train.py --gpus $GPUS --cfg $CFG 
```
- To choose which gpus to use, you can either do ```--gpus 0-7```, or ```--gpus 0,2,4,6```.

For example, you can start with our provided configurations: 

* Train Farneback + ResNet18 + PyBullet for ```predict_friction``` data
```bash
python3 train.py --gpus GPUS --cfg config/predict_friction-farneback-resnet18-pybullet.yaml
```

* Train SpyNet + ResNet18 + PyBullet for ```predict_friction_mass_independent``` data
```bash
python3 train.py --gpus GPUS --cfg config/predict_friction_mass_independent-spynet-resnet18-pybullet.yaml
```

A script is provided to run all configurations used in our experiments
```bash
chmod +x train.sh
./train.sh
```

3. You can also override options in commandline, for example  ```python3 train.py TRAIN.num_epoch 10 ```.

### Testing

To test a model on a folder of various image sequences (```$PATH_IMG```), you can simply do the following:
```bash
python3 -u test.py --imgs $PATH_IMG --gpu $GPU --cfg $CFG --TEST.prediction_timesteps $PREDICTION_TIMESTEPS
```
This will only work if the folder of image sequences is formatted as follows:
```
$PATH_IMG
│   buffer_0_timestep_0.png
│   buffer_0_timestep_1.png
│   buffer_0_timestep_2.png
│   buffer_1_timestep_0.png
│   buffer_1_timestep_1.png
│   buffer_1_timestep_2.png
│   buffer_2_timestep_0.png
│   buffer_2_timestep_1.png
│   buffer_2_timestep_2.png
│   ...
```

The example below is to predict 1, 5, 10 timesteps into the future for each given buffer using Farneback + ResNet18 + PyBullet for ```predict_friction``` data
```bash
python3 eval.py --gpus GPUS --cfg config/predict_friction-farneback-resnet18-pybullet.yaml --TEST.prediction_timesteps [1,5,10]
```

Remember to make sure buffer size is equal to what the model was trained for. By default ```$PATH_IMG = examples```
and will output predictions to folder ```results```.

### Evaluation
1. Evaluate a trained model on the validation set. Evaluations are by default saved in folder ```ckpt```.

For example:

* Evaluate Farneback + ResNet18 + PyBullet for data ```predict_friction``` with future time steps 1, 5, 10
```bash
python3 eval.py --gpus GPUS --cfg config/predict_friction-farneback-resnet18-pybullet.yaml --VAL.prediction_timesteps [1,5,10]
```

* Evaluate SpyNet + ResNet18 + PyBullet for data ```predict_friction_mass_independent``` with future time steps 1, 5, 10
```bash
python3 eval.py --gpus GPUS --cfg config/predict_friction_mass_independent-spynet-resnet18-pybullet.yaml --VAL.prediction_timesteps [1,5,10]
```

A script is provided to run all configurations used in our experiments
```bash
chmod +x eval.sh
./eval.sh
```

## Reference

If you find the code and data generation useful, please cite the following paper and this GitHub Repository:

Learning to See Physics via Visual De-animation. J. Wu, E. Lu, P. Kohli, W. Freeman and J. Tenenbaum. Neural Information Processing Systems
(NeurIPS), 2017.

    [1] @inproceedings{wu2017vda,
             title = {Learning to See Physics via Visual De-animation},
             author = {Wu, Jiajun and Lu, Erika and Kohli, Pushmeet and Freeman, Bill and Tenenbaum, Joshua},
             booktitle = {Advances in Neural Information Processing Systems},
             year = {2017}
        }

    [2] @misc{frennert2023vda,
             title = {An implementation of Visual De-Animation Using PyTorch},
             author = {Jonathan Gustafsson Frennert},
             howpublished = {\url{https://github.com/J0HNN7G/vda}}
             year = {2023},
        }