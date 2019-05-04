import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DoubleConv, self).__init__()
        # [3x3 conv with the 'same' padding, batch norm, relu activation] * 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        outputs = self.conv(x)
        
        return outputs

class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, expansion=1):
        super(InConv, self).__init__()
        if residual:
            layers = []
            dimension_map = None
            if not in_channels == out_channels:
                dimension_map = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                )
            layers.append(BasicBlock(in_channels, out_channels, downsample=dimension_map))
            
            for _ in range(1, expansion):
                layers.append(BasicBlock(out_channels, out_channels, downsample=None))
            
            self.conv = nn.Sequential(*layers)
        else:
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x):
        outputs = self.conv(x)

        return outputs


class DownSamp(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, expansion=1):
        super(DownSamp, self).__init__()
        if residual:
            layers = [nn.MaxPool2d(2)]
            dimension_map = None
            if not in_channels == out_channels:
                dimension_map = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                )
            layers.append(BasicBlock(in_channels, out_channels, downsample=dimension_map))

            for _ in range(1, expansion):
                layers.append(BasicBlock(out_channels, out_channels, downsample=None))
            
            self.down_samp = nn.Sequential(*layers)
        else:
            self.down_samp = nn.Sequential(
                nn.MaxPool2d(2),
                _DoubleConv(in_channels, out_channels)
            )
    
    def forward(self, x):
        outputs = self.down_samp(x)

        return outputs


class UpSamp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, residual=False, expansion=1):
        super(UpSamp, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0, stride=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,\
                kernel_size=4, stride=2, padding=1)
        
        if residual:
            layers = []
            dimension_map = None
            if not in_channels == out_channels:
                dimension_map = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                )
            layers.append(BasicBlock(in_channels, out_channels, downsample=dimension_map))

            for _ in range(1, expansion):
                layers.append(BasicBlock(out_channels, out_channels, downsample=None))
            
            self.conv = nn.Sequential(*layers)
        else:
            self.conv = _DoubleConv(in_channels, out_channels)
        
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1, x2 follow NCHW pattern
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # (left, right, top, bottom), default zero-pad
        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
        diff_y // 2, diff_y - diff_y // 2), mode='constant', value=0)

        # after necessary paddings, x1 and x2 should have the same dimension
        # concatenate them in the second dimension
        x = torch.cat((x1, x2), dim=1)
        outputs = self.conv(x)

        return outputs

class OutConv(nn.Module):
    def __init__(self, in_channels, class_num):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, class_num, kernel_size=1)

    def forward(self, x):
        outputs = self.conv(x)

        return outputs
