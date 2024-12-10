# AAAI2025 - L4DR: LiDAR-4DRadar Fusion for Weather-Robust 3D Object Detection
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.08402)
<div align="center">
  <img src="images/vis.png" width="600"/>
</div>


## :balloon: Introduction
:wave: This is the official repository for **AAAI2025 - L4DR**. 

This repo is also a framework for **LiDAR-based**, **4D radar-based**, **LiDAR-4D radar fusion** based 3D object detection!

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

## 3. Install pcdet
```
python setup.py develop
```

## 4. Install required environment
```
pip install -r requirements.txt
```

## :balloon: Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs) (vod related), 
and the model configs are located within [VoD_models](https://github.com/ylwhxht/L4DR/tree/main/tools/cfgs/VoD_models). 
