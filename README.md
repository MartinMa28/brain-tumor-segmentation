# BraTS2018 brain tumor segmentation

We conduct the segmentation for 3D cases by splitting each 3D case into 2D planes and segmenting those 2D planes using residual UNet.

## Links for BraTS2018 competition

[Home page](https://www.med.upenn.edu/sbia/brats2018.html)

[Data description](https://www.med.upenn.edu/sbia/brats2018/data.html)

[Leaderboard of results on BraTS2018 validation set](https://www.cbica.upenn.edu/BraTS18/lboardValidation.html)

## The structure of residual UNet

We double the number of convolution layers in the original UNet to extract more information from the MRI scans. In order to make the deeper network trainable, we also add residual connection ([He, Kaiming, et al. "Deep residual learning for image recognition."](https://arxiv.org/abs/1512.03385)) to it. The residual UNet used in this repo is called RB UNet 41. "RB" stands for residual blocks, and it has 41 convolutional layers in total.

![Architecture of RB UNet41](/imgs/UNet_ResidualBlock_BraTS.png)

## Results on BraTS2018 validation set

| Network design    | Dice ET | Dice WT | Dice TC |
|:-----------------:|:-------:|:-------:|:-------:|
|Original UNet      |0.6562   |0.8670   |0.6889   |
|RB UNet41          |0.7383   |0.8616   |0.7482   |
|RB UNet41 CLS      |0.7634   |0.8789   |0.8049   |
|RB UNet41 CLS T700 |0.7801   |0.8837   |0.8039   |

RB UNet41 is able to bring out a much better segmentation result than original UNet. In addition to it, after looking at the feedback from UPenn school of medicine, we figure out that RB UNet41 performs really aweful on curtain cases in the validation set, which lowers the average Dice score drastically. While, the median of Dice score looks quite great. So we put a 3D ResNet 50 classifier in front of the segmentation process. Classified HGG/LGG cases are treated respectively by their own segmentation model. "RB UNet41 CLS" shows the result of our classfication + segmentation pipeline. Even if we added the classifier, some cases' Dice score of ET region is still pretty low. And we find that those bad cases' predicted segmentation result only has very limited pixels be classified as the ET region. So we set a threshold (700) for the minimum number of ET pixels, if the ET pixels are less than 700, they will be classified into the background class.

![Flowchat of the entire pipeline](/imgs/MRI_segmentation_flowchart.png)