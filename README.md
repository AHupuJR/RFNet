# RFNet: Real-time Fusion Network for RGB-D Semantic Segmentation Incorporating Unexpected Obstacle Detection of Road-driving Images
This repository is a Pytorch implementation for  

>Sun, Lei, et al. "Real-time Fusion Network for RGB-D Semantic Segmentation Incorporating Unexpected Obstacle Detection for Road-driving Images." arXiv preprint arXiv:2002.10570 (2020).

If you want to use this code in your research, please cite the [paper](https://arxiv.org/abs/2002.10570).


## Requirement
    Python 3.6  
    Pytorch 1.1  
    Torchvision 0.3  
    Opencv 3.3.1

## Datasets
Get dataset from [Cityscapes](https://www.cityscapes-dataset.com/), and from [Lost and Found](http://www.6d-vision.com/lostandfounddataset).  

If you want to use multi-dataset training, mix two datasets and the directory structure should be like this:
    
    ├─disparity
    │  ├─test
    │  │  ├─berlin
    │  │  ├─bielefeld
    │  │  ├─bonn
    │  │  ├─...
    │  │  └─munich
    │  ├─train
    │  │  ├─01_Hanns_Klemm_Str_45
    │  │  ├─03_Hanns_Klemm_Str_19
    │  │  ├─...
    │  │  └─zurich
    │  └─val
    │      ├─02_Hanns_Klemm_Str_44
    │      ├─04_Maurener_Weg_8
    │      ├─05_Schafgasse_1
    │      ├─...
    │      └─munster
    ├─gtFine
    │  ├─train
    │  │  ├─01_Hanns_Klemm_Str_45
    │  │  ├─03_Hanns_Klemm_Str_19
    │  │  ├─...
    │  │  └─zurich
    │  └─val
    │      ├─02_Hanns_Klemm_Str_44
    │      ├─04_Maurener_Weg_8
    │      ├─05_Schafgasse_1
    │      ├─...
    │      └─munster
    └─leftImg8bit
        ├─test
        │  ├─berlin
        │  ├─bielefeld
        │  ├─bonn
        │  ├─...
        │  └─munich
        ├─train
        │  ├─01_Hanns_Klemm_Str_45
        │  ├─03_Hanns_Klemm_Str_19
        │  ├─...
        │  └─zurich
        └─val
            ├─02_Hanns_Klemm_Str_44
            ├─04_Maurener_Weg_8
            ├─05_Schafgasse_1
            ├─...
            └─munster

## Pretrained weights
### BaiduNetdisk
Download weights for [Cityscapes](https://pan.baidu.com/s/1scfm_PQL6v3DiMikMDePkA), password: 4lts  
Download weights for Multi-dataset: [Cityscapes and Lost and Found](https://pan.baidu.com/s/1KjhafvJC5zHWsnenok50OQ), password: t1mh

### Google Drive
Download weights for [Cityscapes](https://drive.google.com/file/d/1nJkYqvQv6BSTciyXHqe66fTzgWe4alr1/view?usp=sharing)    
Download weights for Multi-dataset: [Cityscapes and Lost and Found](https://drive.google.com/file/d/1aKa1hb6nVUSd1Gz6NQXk8W03WpUpw3EC/view?usp=sharing)

## Training
Edit path to your dataset in `mypath.py`.

`python train.py --depth --lr 1e-4 --weight-decay 2.5e-5 --workers 4 --epochs 200 --batch-size 8 --val-batch-size 3 --gpu-ids 0 --checkname test --eval-interval 2 --dataset citylostfound --loss-type ce --use-balanced-weights`

## Evaluation
`python eval.py --dataset citylostfound --weight-path your/path/to/weight/ --depth`

*Note: The code is partially based [Swiftnet](https://github.com/orsic/swiftnet) and [DeepLab v3+](https://github.com/jfzhang95/pytorch-deeplab-xception).*