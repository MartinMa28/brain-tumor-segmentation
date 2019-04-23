# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torch.utils.model_zoo as model_zoo

class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCN8sScaledBN(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        # x4 and x3 are scaled by the factor of 0.01 and 0.0001 respectively
        score = self.relu(self.bn1(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4 * 0.01                         # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.bn2(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3 * 0.0001                       # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.bn3(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.relu(self.bn4(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.relu(self.bn5(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCN8sScaled(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)


    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        # x4 and x3 are scaled by the factor of 0.01 and 0.0001 respectively
        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = score + x4 * 0.01                         # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = score + x3 * 0.0001                       # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv3(score))            # size=(N, 128, x.H/4, x.W/4)
        score = self.relu(self.deconv4(score))            # size=(N, 64, x.H/2, x.W/2)
        score = self.relu(self.deconv5(score))            # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCN8sScaledOGBN(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, n_class, kernel_size=4, stride=2, padding=1)
        self.conv_x4 = nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0)
        self.bn1     = nn.BatchNorm2d(n_class)
        self.deconv2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1)
        self.bn2     = nn.BatchNorm2d(n_class)
        self.conv_x3 = nn.Conv2d(256, n_class, kernel_size=1, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, padding=4)
        self.bn3     = nn.BatchNorm2d(n_class)
        
    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        # x4 and x3 are scaled by the factor of 0.01 and 0.0001 respectively
        score = self.relu(self.bn1(self.deconv1(x5)))
        score = score + self.conv_x4(x4 * 0.01)
        score = self.relu(self.bn2(self.deconv2(score)))
        score = score + self.conv_x3(x3 * 0.0001)
        score = self.relu(self.bn3(self.deconv3(score)))

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCN8sScaledOG(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, n_class, kernel_size=4, stride=2, padding=1)
        self.conv_x4 = nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1)
        self.conv_x3 = nn.Conv2d(256, n_class, kernel_size=1, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, padding=4)
    
        
    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        # x4 and x3 are scaled by the factor of 0.01 and 0.0001 respectively
        score = self.relu(self.deconv1(x5))
        score = score + self.conv_x4(x4 * 0.01)
        score = self.relu(self.deconv2(score))
        score = score + self.conv_x3(x3 * 0.0001)
        score = self.relu(self.deconv3(score))

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCN8s_bilinear(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, padding=0, stride=1)
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        # x4 and x3 are scaled by the factor of 0.01 and 0.0001 respectively
        score = nn.functional.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
        score += x4 * 0.01
        score = nn.functional.interpolate(score, scale_factor=2, mode='bilinear', align_corners=True)
        score = self.relu(self.conv1(score))
        score += x3 * 0.0001
        score = nn.functional.interpolate(score, scale_factor=8, mode='bilinear', align_corners=True)
        score = self.relu(self.conv2(score))

        score = self.classifier(score)

        # size=(N, n_class, x.H/1, x.W/1)
        return score

class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False, batch_norm=False):
        super().__init__(make_layers(cfg[model], batch_norm=batch_norm))
        self.ranges = ranges
        self.model_urls = {
            'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        }
        if batch_norm:
            self.model = 'vgg16_bn'
        else:
            self.model = 'vgg16'

        if pretrained:
            #exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
            self.load_state_dict(model_zoo.load_url(self.model_urls[self.model]))

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges[self.model])):
            for layer in range(self.ranges[self.model][idx][0], self.ranges[self.model][idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg16_bn': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    batch_size, n_class, h, w = 2, 20, 512, 512

    # test output size
    vgg_model = VGGNet(requires_grad=True, batch_norm=True)
    inp = torch.autograd.Variable(torch.randn(batch_size, 3, 512, 512))
    output = vgg_model(inp)
    assert output['x5'].size() == torch.Size([batch_size, 512, 16, 16])

    # fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)
    # input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    # output = fcn_model(input)
    # assert output.size() == torch.Size([batch_size, n_class, h, w])

    # fcn_model = FCN16s(pretrained_net=vgg_model, n_class=n_class)
    # input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    # output = fcn_model(input)
    # assert output.size() == torch.Size([batch_size, n_class, h, w])

    # fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)
    # input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    # output = fcn_model(input)
    # assert output.size() == torch.Size([batch_size, n_class, h, w])

    # fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    # input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    # output = fcn_model(input)
    # assert output.size() == torch.Size([batch_size, n_class, h, w])

    # fcn_model = FCN8s_bilinear(pretrained_net=vgg_model, n_class=n_class)
    # inp = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    # ouput = fcn_model(inp)
    # assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN8sScaledOGBN(pretrained_net=vgg_model, n_class=n_class)
    inp = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(inp)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    # fcn_model = FCN8sScaledOG(pretrained_net=vgg_model, n_class=n_class)
    # inp = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    # output = fcn_model(inp)
    # assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN8sScaledBN(pretrained_net=vgg_model, n_class=n_class)
    inp = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(inp)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    # fcn_model = FCN8sScaled(pretrained_net=vgg_model, n_class=n_class)
    # inp = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    # output = fcn_model(inp)
    # assert output.size() == torch.Size([batch_size, n_class, h, w])
    print("Pass size check")

    # test a random batch, loss should decrease
    fcn_model = FCN8sScaledOGBN(pretrained_net=vgg_model, n_class=n_class)
    criterion = nn.BCELoss()
    # optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(fcn_model.parameters(), lr=1e-4)
    inp = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    y = torch.autograd.Variable(torch.randn(batch_size, n_class, h, w), requires_grad=False)
    for iter in range(10):
        optimizer.zero_grad()
        output = fcn_model(inp)
        output = torch.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        print("iter{}, loss {}".format(iter, loss.data.item()))
        optimizer.step()
