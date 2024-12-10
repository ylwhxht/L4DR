# AAAI2025 - L4DR: LiDAR-4DRadar Fusion for Weather-Robust 3D Object Detection
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.08402)
<div align="center">
  <img src="images/vis.png" width="600"/>
</div>


## :balloon: Introduction
:wave: This is the official repository for **AAAI2025 - L4DR**. 

This repo is also a framework for **LiDAR-based**, **4D radar-based**, **LiDAR-4D radar fusion** based 3D object detection for VoD dataset!

## :balloon: Installation

This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 

### 1. Clone (or download) the source code 
```
git clone https://github.com/ylwhxht/L4DR.git 
cd L4DR
```
 
### 2. Create conda environment and set up the base dependencies
```
conda create --name l4dr python=3.7 cmake=3.22.1
conda activate l4dr
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install spconv-cu113
```

### 3. Install pcdet
```
python setup.py develop
```

### 4. Install required environment
```
pip install -r requirements.txt
```

## :balloon: Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs) (vod related), 
and the model configs are located within [VoD_models](https://github.com/ylwhxht/L4DR/tree/main/tools/cfgs/VoD_models). 


### Dataset Preparation
#### 1. Dataset download
Please follow [VoD Dataset](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/docs/GETTING_STARTED.md) to download dataset.

* (Optional)
If you want to reproduce our fog simulation-related experiments, you need to run [fog simulation](https://github.com/MartinHahner/LiDAR_fog_sim) on the VoD lidar point cloud. The relevant parts of fog simulation only need to refer to their code and configuration. 
**This part may be difficult and complex. If you need it, you can contact me to request our VoD dataset after fog simulation.**

After the preparation, the format of how the dataset is provided:

```
View-of-Delft-Dataset (root)
    ├── lidar (kitti dataset where velodyne contains the LiDAR point clouds)
      ...
    ├── radar (kitti dataset where velodyne contains the radar point clouds)
      ...
    ├── radar_3_scans (kitti dataset where velodyne contains the accumulated radar point clouds of 3 scans)
      ...
    ├── radar_5_scans (kitti dataset where velodyne contains the accumulated radar point clouds of 5 scans)
      ...
    ├── fog_sim_lidar (unique data, here was obtained through fog simulation)
      ...
```


#### 2. Data structure alignment
In order to train the LiDAR and 4DRadar fusion model according to the logic of OpenPCDet, we then need to generate LiDAR and 4DRadar fusion data infos.
* First, create an additional folder with lidar and radar point clouds in the VoD dataset directory (here we call it **rlfusion_5f**):
```
View-of-Delft-Dataset (root)
    ├── lidar
    ├── radar
    ├── radar_3_scans
    ├── radar_5_scans
    ├── rlfusion_5f (mainly used)
    ├── fog_sim_lidar 
```

* Then, refer to the following structure to place the corresponding files in the rlfusion_5f folder:
```
rlfusion_5f
    │── ImageSets
    │── training
       ├── calib (lidar_calib)
       ├── image_2
       ├── label_2
       ├── lidar (lidar velodyne)
       ├── lidar_calib
       ├── pose
       ├── radar (single frame radar velodyne)
       ├── radar_5f (radar_5_scans velodyne)
       ├── radar_calib
    │── testing
       ... like training (except label)
```


#### 3. Data infos generation
* Firstly, remember to change **DATA-PATH** in the [VoD dataset cfg file](https://github.com/ylwhxht/L4DR/blob/main/tools/cfgs/dataset_configs/Vod_fusion.yaml).

* Generate the data infos by running the following command: 
```
python -m pcdet.datasets.vod.vod_dataset create_vod_infos tools/cfgs/dataset_configs/Vod_fusion.yaml
```

### Training & Testing


#### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```

For example
```shell script
CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_train.sh 2 --cfg_file cfgs/VoD_models/L4DR.yaml --extra_tag 'l4dr_demo' --sync_bn
```

#### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

For example

```shell script
CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_test.sh 2 --cfg_file cfgs/VoD_models/L4DR.yaml --extra_tag 'l4dr_demo' --ckpt /mnt/32THHD/hx/Outputs/output/VoD_models/PP_DF_OurGF/mf2048_re/ckpt/checkpoint_epoch_100.pth
```
