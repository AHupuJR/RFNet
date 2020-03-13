# RFNet
RFNet: Real-time Fusion Network for RGB-D Semantic Segmentation Incorporating Unexpected Obstacle Detection of Road-driving Images

# Requirement
Python 3.6  
Pytorch 1.1  
Torchvision 0.3  
Opencv 3.3.1

# Pretrained weights
Download weights for [Cityscapes](https://pan.baidu.com/s/1m_gen0k1VZhyMSAzdPLiSw), password:86gs
Download weights for Multi-dataset: [Cityscapes and Lost and Found](https://pan.baidu.com/s/14N6Vybu0cTiBOEycpemysQ), password:ypmc

# Evaluation
python eval.py --dataset citylostfound --weight-path your/path/to/weight/ --depth
