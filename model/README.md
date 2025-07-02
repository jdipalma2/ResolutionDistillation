Modified version of [torchvision ResNet module](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) to replace Batch Normalization layers with Group Normalization ones. 

This is to increase compatibility with gradient accumulation. Batch normalization doesn't work well with small batch sizes, while group normalization does.