import torch
import torchvision
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, conv1x1, conv3x3, Bottleneck, model_urls
import torch.nn as nn
import torch.optim as optim


class ResNetEncoder(ResNet):
    def __init__(self, input_channels, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetEncoder, self).__init__(block, layers, num_classes, zero_init_residual)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)


def resnet50_encoder_factory(input_channels, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        input_channels: the number of input channels
    """
    model = ResNetEncoder(input_channels, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_inputs, n_classes=2, pre_trained=False):
        """
        arguments:
            n_inputs: number of input channels
            n_classes: number of output channels
            pre_trained: whether load the pre-trained model
        """
        super().__init__()
        resnet = resnet50_encoder_factory(n_inputs, pretrained=pre_trained)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + n_inputs, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools["layer_0"] = x
        x = self.input_block(x)
        pre_pools["layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools["layer_{}".format(i)] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = "layer_{}".format(UNetWithResnet50Encoder.DEPTH - 1 - i)
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    
    def dice_coef(self, preds, targets):
        smooth = 5e-3
        num = preds.size(0)              # batch size
        preds_flat = preds.view(num, -1).float()
        targets_flat = targets.view(num, -1).float()

        intersection = (preds_flat * targets_flat).sum()

        return (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        score = self.dice_coef(probs, targets)
        score = 1 - score

        return score

if __name__ == "__main__":
    batch_size, n_channels, height, width = 4, 2, 256, 256
    num_classes = 1
    inputs = torch.randn(batch_size, n_channels, height, width)
    unet_resnet = UNetWithResnet50Encoder(n_channels, n_classes=num_classes)
    outputs = unet_resnet(inputs)
    assert outputs.size() == torch.Size([batch_size, num_classes, height, width])
    print('pass the dimension check')

    criterion = SoftDiceLoss()
    optimizer = optim.Adam(unet_resnet.parameters(), lr=1e-4)
    inputs = torch.randn(batch_size, n_channels, height, width)
    targets = torch.randint(low=0, high=2, size=(batch_size, height, width))

    for iter in range(10):
        optimizer.zero_grad()
        outputs = unet_resnet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        print('iter{}, loss={}'.format(iter, loss))
        optimizer.step()
