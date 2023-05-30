# Occ-BEV: Multi-Camera Unified Pre-training via 3D Scene Reconstruction 
(for BEVFormer, BEVDet, BEVDepth and Occupancy prediction)

> [Paper in arXiv](http://arxiv.org/abs/xxx) 

# Abstract
Multi-camera 3D perception has emerged as a prominent research field in autonomous driving, offering a viable and cost-effective alternative to LiDAR- based solutions. However, existing multi-camera algorithms primarily rely on monocular image pre-training, which overlooks the spatial and temporal correlations among different camera views. To address this limitation, we propose the first multi-camera unified pre-training framework called Occ-BEV, which involves initially reconstructing the 3D scene as the foundational stage and subsequently fine-tuning the model on downstream tasks. Specifically, a 3D decoder is designed for leveraging Bird’s Eye View (BEV) features from multi-view images to predict the 3D geometry occupancy to enable the model to capture a more comprehensive understanding of the 3D environment. One significant advantage of Occ-BEV is that it can utilize a vast amount of unlabeled image-LiDAR pairs for pre-training. The proposed multi-camera unified pre-training framework demonstrates promising results in key tasks such as multi-camera 3D object detection and semantic scene completion. When compared to monocular pre-training methods on the nuScenes dataset, Occ-BEV demonstrates a significant improvement of 2.0% in mAP and 2.0% in NDS for 3D object detection, as well as a 0.8% increase in mIOU for semantic scene completion.


# Methods
![method](docs/flowchart.png "model arch")


# Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/getting_started.md)

# Model Zoo

| Backbone | Method | Pre-training | Lr Schd | NDS| mAP| Config |
| :---: | :---: | :---: | :---: | :---:| :---: | :---: |
| R101-DCN  | [BEVFormer](https://github.com/fundamentalvision/BEVFormer) | FCOS3D | 24ep | 51.7 | 41.6 | [config](BEVFormer/projects/configs/bevformer/bevformer_base.py)/[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth) |
| R101-DCN  | BEVFormer | Occ-BEV | 24ep | 53.4 |43.8 |[config](projects/configs/bevformer/occ_bev_sweep2.py)/[pre-trained model](https://drive.google.com/file/d/1tXylQhYLAH6c-gAJD0dUeZxwOPUD4rZX/view?usp=drive_link)/[log](https://drive.google.com/file/d/1ignosErdLqiRdvSqEon7P7cHWCYGufQN/view?usp=drive_link)|


# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{occ-bev,
  title={Occ-BEV: Multi-Camera Unified Pre-training via 3D Scene Reconstruction},
  author={Chen Min, Xinli Xu, Dawei Zhao, Liang Xiao, Yiming Nie, and Bin Dai}
  journal={arXiv preprint},
  year={2023}
}
```

# Acknowledgement

Many thanks to these excellent open source projects:
- [Occ3D](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) 

- [occupancy-for-nuscenes](https://github.com/Megvii-BaseDetection/BEVDepth)

- [DETR3D](https://github.com/WangYueFt/detr3d) 

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) 

- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)

- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

  
