import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
import torch.utils.model_zoo as model_zoo

# global helper variables for VGG encoder from torchvision.models.vgg
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# PyTorch vgg helper function
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
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


# vgg16 encoder
class VGGEncoder(VGG):
    def __init__(self,\
        pretrained=True,\
        model_type='vgg16',\
        requires_grad=True,\
        remove_fc=True, show_params=False):
        super(VGGEncoder, self).__init__(make_layers(cfg[model_type]))
        self.type = model_type
        self.ranges = ranges[self.type]

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls[self.type]))

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:
            # delete vgg's fully connected layers
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())
        
    
    def forward(self, x):
        outputs = {}

        for idx in range(len(self.ranges)):
            # idx: index for Max-Pooling layers
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            outputs['x{}'.format(idx + 1)] = x
        
        return outputs




