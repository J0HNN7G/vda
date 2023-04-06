# Visual De-Animation in PyTorch for Pool Balls

This is an unofficial PyTorch implementation of visual de-animation.

## Supported models
We split our models into optical flow, perceptual module, physics module and graphics module. We have provided some pre-configured models in the ```config``` folder.

Optical Flow:
- Farneback
- SpyNet

Perceptual Module:
- Resnet18
- LeNet

Physics Module:
- PyBullet

Graphics Module:
- PyBullet (OpenGL)

## Performance:

<table><tbody>
    <th valign="bottom">Opt. Flow</th>
    <th valign="bottom">Network</th>
    <th valign="bottom">Physics</th>
    <th valign="bottom">Graphics</th>
    <th valign="bottom">MSE (px)</th>
    <th valign="bottom">Pos. Pred. (px)</th>
    <th valign="bottom">Vel. Pred. (px)</th>
    <th valign="bottom">Inf. Speed (fps)</th>
    <tr>
        <td>Farneback</td>
        <td>ResNet18</td>
        <td>PyBullet</td>
        <td>PyBullet</td>
        <td>TODO</td>
        <td>TODO</td>
        <td>TODO</td>
        <td>TODO</td>
    </tr>
    <tr>
        <td>SpyNet</td>
        <td>ResNet18</td>
        <td>PyBullet</td>
        <td>PyBullet</td>
        <td>TODO</td>
        <td>TODO</td>
        <td>TODO</td>
        <td>TODO</td>
    </tr> 
    <tr>
        <td>Farneback</td>
        <td>LeNet</td>
        <td>PyBullet</td>
        <td>PyBullet</td>
        <td>TODO</td>
        <td>TODO</td>
        <td>TODO</td>
        <td>TODO</td>
    </tr> 
    <tr>
        <td>SpyNet</td>
        <td>LeNet</td>
        <td>PyBullet</td>
        <td>PyBullet</td>
        <td>TODO</td>
        <td>TODO</td>
        <td>TODO</td>
        <td>TODO</td>
    </tr> 


</tbody></table>

The training is benchmarked with a single Nvidia GTX 1060  (6GB GPU Memory), the inference speed is benchmarked without visualization.

## Environment
The code is developed under the following configurations.
- Hardware: >=1 GPUs for training, >=1 GPU for testing (set ```[--gpus GPUS]``` accordingly)
- Software: Ubuntu 20.04.6 LTS, ***CUDA=11, Python=3.8.10, PyTorch=1.13.1***
- Dependencies: numpy, pybullet, torch, torchvision, Pillow, opencv-python, pandas, matplotlib, tqdm, yacs

## Training
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

## Testing

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

## Evaluation
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

## Integration with other projects
This library can be installed via `pip` to easily integrate with another codebase
```bash
pip install git+https://github.com/J0HNN7G/intuitive_physics.git
```

Now this library can easily be consumed programmatically. For example
```python
from vda.config import cfg
from vda.dataset import TestDataset
from vda.models import ModelBuilder, PerceptualModule
```

## Reference

If you find the code or data generation useful, please link to this GitHub repository and cite the following papers:

Learning to See Physics via Visual De-animation. J. Wu, E. Lu, P. Kohli, W. Freeman and J. Tenenbaum. Neural Information Processing Systems
(NeurIPS), 2017.

    @inproceedings{wu2017vda,
         title = {Learning to See Physics via Visual De-animation},
         author = {Wu, Jiajun and Lu, Erika and Kohli, Pushmeet and Freeman, Bill and Tenenbaum, Joshua},
         booktitle = {Advances in Neural Information Processing Systems},
         year = {2017}
    }