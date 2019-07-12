# BraTS2018 brain tumor segmentation

We conduct the segmentation for 3D cases by splitting each 3D case into 2D planes and segmenting those 2D planes using residual UNet.

## Links for BraTS2018 competition

[Home page](https://www.med.upenn.edu/sbia/brats2018.html)

[Data description](https://www.med.upenn.edu/sbia/brats2018/data.html)

[Leaderboard of results on BraTS2018 validation set](https://www.cbica.upenn.edu/BraTS18/lboardValidation.html)

## The structure of residual UNet

We double the number of convolution layers in the original UNet to extract more information from the MRI scans. In order to make the deeper network trainable, we also add residual connection ([He, Kaiming, et al. "Deep residual learning for image recognition."](https://arxiv.org/abs/1512.03385)) to it. The residual UNet used in this repo is called RB UNet 41. "RB" stands for residual blocks, and it has 41 convolutional layers in total.

![Architecture of RB UNet41](/imgs/UNet_ResidualBlock_BraTS.png)