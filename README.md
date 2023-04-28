# Baseline Methods for SuctionNet-1Billion

Baseline methods in RA-L paper "SuctionNet-1Billion:  A  Large-Scale  Benchmark  for  Suction  Grasping" 

![Image Title](https://github.com/graspnet/suctionnet-baseline/blob/master/framework3.jpg)

## Dataset

Download data and labels from our [SuctionNet webpage](https://graspnet.net/suction).

## Environment

The code has been tested with `CUDA 10.1` and `pytorch 1.4.0` on ubuntu `16.04`

## Training Prerequisites

To train the networks, you need additional labels including 2D mappings of seal labels, bounding boxes of objects and object centers.

change the directory to `neural_network`

```
cd neural_network
```

To generate the 2D mappings of seal label, run the following command:

```
python score_mapping.py \
--dataset_root /path/to/SuctionNet/dataset \
--saveroot /path/to/save/additional/labels \
--camera realsense \ # kinect or realsense
--sigma 4 \	# sigmal of the 2D gaussian kernel to map the score
--pool_size 10 \ # number of cpu threads to use
--save_visu # whether to save visualizations
```

or modify [scripts/score_mapping.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/score_mapping.sh) and run `sh scripts/score_mapping.sh`.

To get bounding boxes and centers of the objects, run the following command:

```
python cal_center_bbox.py \
--dataset_root /path/to/SuctionNet/dataset \
--saveroot /path/to/save/additional/labels \
--camera realsense \ # kinect or realsense
--pool_size 10 \ # number of cpu threads to use
--save_visu # whether to save visualizations
```

or modify the [scripts/cal_center_bbox.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/cal_center_bbox.sh) and run `sh scripts/cal_center_bbox.sh`.

Please make sure the `--saveroot` args are the same for the above two commands.

Note that the 2D mappings of seal label can take up to `177 G` disk space. We save them in advance to make the training process more efficient. You may also modify the mapping to an online version but this will be much slower for training.

## Usage

### Neural Networks

Change the directory to `neural_network`:

```
cd neural_network
```

For training, use the following command:

```
python train.py \
--model model_name \ 
--camera realsense \ # realsense or kinect
--log_dir /path/to/save/the/model/weights \
--data_root /path/to/SuctionNet/dataset \
--label_root /path/to/the/additional/labels \
--batch_size 8
```

or modify [scripts/deeplabv3plus_train.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_train.sh), [scripts/deeplabv3plus_train_depth.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference_depth.sh), [scripts/convnet_train.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/convnet_train.sh) for training our RGB-D model, depth model and fully convolutional network (FCN) model.

For inference, use the following command: 

```
python inference.py \
--model model_name \
--checkpoint_path /path/to/the/saved/model/weights \
--split test_seen \ # can be test, test_seen, test_similar, test_novel
--camera realsense \ # realsense or kinect
--dataset_root /path/to/SuctionNet/dataset \
--save_dir /path/to/save/the/inference/results \
--save_visu # whether to save the visualizations
```

or modify [scripts/deeplabv3plus_inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference.sh), [scripts/deeplabv3plus_inference_depth.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference_depth.sh), [scripts/convnet_inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/convnet_inference.sh) to inference with our RGB-D model, depth model and fully convolutional network (FCN) model.

### Normal STD

Change the directory to `normal_std` by:

```
cd normal_std
```

This method does not need to train, you can directly inference with the following command:

```
python inference.py 
--split test_seen \ # can be test, test_seen, test_similar, test_novel
--camera realsense \ # realsense or kinect
--save_root /path/to/save/the/inference/results \
--dataset_root /path/to/SuctionNet/dataset \
--save_visu
```

or modify [inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/normal_std/inference.sh) and run `sh inference.sh`

## Pre-trained Models

### RGB-D Models

We provide models including [our model for realsense](https://drive.google.com/file/d/18TbctdhpNXEKLYDWFzI9cT1Wnhe-tn9h/view?usp=sharing), [our model for kinect](https://drive.google.com/file/d/1gOz_KmIugBGUtpcyHAgYO01T0h5ZqOl9/view?usp=sharing), [Fully Conv Net for realsense](https://drive.google.com/file/d/1hgYYIvw5Xy-r5C8IitKizswtuMV_EqPP/view?usp=sharing) ,[Fully Conv Net for kinect](https://drive.google.com/file/d/1A6K5EmItBuDaxrWyz5g8zSHY5Kw1_NnX/view?usp=sharing).

### Depth Models

Our models only taking in depth images are also provided [for realsense](https://drive.google.com/file/d/1q2W2AV663PNT4_TYo5zZtYxjenZJ7GAb/view?usp=sharing) and [for kinect](https://drive.google.com/file/d/1mAzFC9dlEDBuoHQp7JGTcTkKGSwFnVth/view?usp=sharing).

## Citation

if you find our work useful, please cite

```
@ARTICLE{suctionnet,
  author={Cao, Hanwen and Fang, Hao-Shu and Liu, Wenhai and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping}, 
  year={2021},
  volume={6},
  number={4},
  pages={8718-8725},
  doi={10.1109/LRA.2021.3115406}}
```


# 使用记录
## 推理记录
- 下载预训练模型：[realsense预训练模型](https://drive.google.com/file/d/1q2W2AV663PNT4_TYo5zZtYxjenZJ7GAb/view), [kinect预训练模型](https://drive.google.com/file/d/1mAzFC9dlEDBuoHQp7JGTcTkKGSwFnVth/view)
- 下载数据集：在[graspnet网址](https://graspnet.net/datasets.html)中下载"Test Images"

## 报错以及解决方法
报错：
```bash
    from torchvision.models.utils import load_state_dict_from_url
ModuleNotFoundError: No module named 'torchvision.models.utils'
```
解决：`torchvision.models.utils import load_state_dict_from_url`这个用法在比较低和最新的torchvision版本中都不适用，可以更改为如下：
```python
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
```
另一种解决方法是：将torchvision的版本改为

---

报错：
```bash
Traceback (most recent call last):
  File "/home/kb/suctionnet-baseline/neural_network/inference_phoxi.py", line 88, in <module>
    net.load_state_dict(checkpoint['model_state_dict'])
  File "/home/kb/anaconda3/envs/ceshi/lib/python3.8/site-packages/torch/nn/modules/module.py", line 2041, in load_state_dict
    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
RuntimeError: Error(s) in loading state_dict for DataParallel:
	~~size mismatch for module.backbone.conv1.weight: copying a param with shape torch.Size([64, 1, 7, 7]) from checkpoint, the shape in current model is torch.Size([64, 4, 7, 7]).~~
```

记录：尝试进行pytorch和torchvision降级，看是否可以解决问题。  \
我当前的版本如下：  
```buildoutcfg
torch                  2.0.0
torchaudio             2.0.0
torchvision            0.15.0
```
项目readme中使用的版本如下，注意我用的是python3.7：
```buildoutcfg
The code has been tested with CUDA 10.1 and pytorch 1.4.0 on ubuntu 16.04
```
```buildoutcfg
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```