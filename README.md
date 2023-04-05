# Visual De-Animation in PyTorch for Pool Balls

This is a PyTorch implementation of visual de-animation.

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
        <td>0.0008</td>
        <td>9.0000</td>
        <td>0.5000</td>
        <td>3.3000</td>
    </tr>
    <tr>
        <td>SpyNet</td>
        <td>ResNet18</td>
        <td>PyBullet</td>
        <td>PyBullet</td>
        <td>0.0008</td>
        <td>9.0000</td>
        <td>0.5000</td>
        <td>3.3000</td>
    </tr>   


</tbody></table>

The training is benchmarked with a single Nvidia GTX 1060  (6GB GPU Memory), the inference speed is benchmarked without visualization.

## Environment
The code is developed under the following configurations.
- Hardware: >=1 GPUs for training, >=1 GPU for testing (set ```[--gpus GPUS]``` accordingly)
- Software: Ubuntu 20.04.6 LTS, ***CUDA=11, Python=3.8.10, PyTorch=1.13.1***
- Dependencies: numpy, opencv, matplotlib, yacs, tqdm

## Quick start: Test on an image using our trained model 
1. Here is a simple demo to do inference on a single image:
```bash
chmod +x demo_test.sh
./demo_test.sh
```
This script downloads a trained model (ResNet50dilated + PPM_deepsup) and a test image, runs the test script, and saves predicted segmentation (.png) to the working directory.

2. To test on an image or a folder of images (```$PATH_IMG```), you can simply do the following:
```
python3 -u test.py --imgs $PATH_IMG --gpu $GPU --cfg $CFG
```

## Training
1. Download the ADE20K scene parsing dataset:
```bash
chmod +x download_ADE20K.sh
./download_ADE20K.sh
```
2. Train a model by selecting the GPUs (```$GPUS```) and configuration file (```$CFG```) to use. During training, checkpoints by default are saved in folder ```ckpt```.
```bash
python3 train.py --gpus $GPUS --cfg $CFG 
```
- To choose which gpus to use, you can either do ```--gpus 0-7```, or ```--gpus 0,2,4,6```.

For example, you can start with our provided configurations: 

* Train MobileNetV2dilated + C1_deepsup
```bash
python3 train.py --gpus GPUS --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml
```

* Train ResNet50dilated + PPM_deepsup
```bash
python3 train.py --gpus GPUS --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml
```

* Train UPerNet101
```bash
python3 train.py --gpus GPUS --cfg config/ade20k-resnet101-upernet.yaml
```

3. You can also override options in commandline, for example  ```python3 train.py TRAIN.num_epoch 10 ```.


## Evaluation
1. Evaluate a trained model on the validation set. Add ```VAL.visualize True``` in argument to output visualizations as shown in teaser.

For example:

* Evaluate MobileNetV2dilated + C1_deepsup
```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml
```

* Evaluate ResNet50dilated + PPM_deepsup
```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml
```

* Evaluate UPerNet101
```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-resnet101-upernet.yaml
```

## Integration with other projects
This library can be installed via `pip` to easily integrate with another codebase
```bash
pip install git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master
```

Now this library can easily be consumed programmatically. For example
```python
from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
```

## Reference

If you find the code or pre-trained models useful, please cite the following papers:

Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV), 2018. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2018semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={International Journal on Computer Vision},
      year={2018}
    }

Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    