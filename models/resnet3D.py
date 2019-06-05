import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)

def conv1x1x1(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1,
                    stride=stride, padding=0, bias=False)


class BasicBlock3D(BasicBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

        
class Bottleneck3D(Bottleneck):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = conv1x1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


class ResNet3D(ResNet):
    # override parent (ResNet)'s init
    def __init__(self, block, layers, num_classes=4, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv3d(4, 32, kernel_size=3, stride=2, padding=1,
                            bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
    def _make_layer(self, block, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)


def resnet34_3D(**kwargs):
    """
    Constructs a ResNet34 3D model.
    """
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], **kwargs)


def resnet50_3D(**kwargs):
    """
    Constructs a ResNet50 3D model.
    """
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)


if __name__ == "__main__":
    batch_size, n_channels, height, width, depth = 1, 4, 160, 160, 144
    num_classes = 2
    inputs = torch.randn(batch_size, n_channels, height, width, depth)
    model = resnet50_3D(num_classes=num_classes)
    outputs = model(inputs)
    assert outputs.size() == torch.Size([batch_size, num_classes])
    print('pass the dimension check')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.randint(low=0, high=2, size=(batch_size,))
    
    for iter in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        print('iter{}, loss{}'.format(iter, loss))
        optimizer.step()
