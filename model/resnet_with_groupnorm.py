import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=1):
    """3x3 convolution with padding
    Unchanged from original torchvision implementation
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=padding,
                     groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, padding=0):
    """1x1 convolution
    Unchanged from original torchvision implementation
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False,
                     padding=padding)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, padding=(1, 1), downsample=None, num_groups=32):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv_1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride, padding=padding[0],
                              dilation=dilation)
        self.norm_1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(in_planes=planes, out_planes=planes, padding=padding[1], dilation=dilation)
        self.norm_2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.norm_2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, layers=(1, 1, 1, 1), planes=(32, 64, 128, 256), num_classes=1000, dilation=1, num_groups=32):
        """
        Modified slightly from original torchvision implementation to replace batch with group normalization
        """
        super(ResNet, self).__init__()
        self._norm_layer = nn.GroupNorm
        self.dilation = dilation
        self.num_groups = num_groups

        self.planes = planes
        self.in_planes = self.planes[0]
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2,
                                padding=(3 * self.dilation), bias=False, dilation=self.dilation)
        self.norm_1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        dilation_padding = ((self.dilation, self.dilation), (self.dilation, self.dilation))

        self.layer_1 = self._make_layer(planes=self.planes[0], blocks=layers[0], padding=dilation_padding)
        self.layer_2 = self._make_layer(planes=self.planes[1], blocks=layers[1], stride=2, padding=dilation_padding)
        self.layer_3 = self._make_layer(planes=self.planes[2], blocks=layers[2], stride=2, padding=dilation_padding)
        self.layer_4 = self._make_layer(planes=self.planes[3], blocks=layers[3], stride=2, padding=dilation_padding)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=self.planes[3] * BasicBlock.expansion, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

    def _make_layer(self, planes, blocks, stride=1, padding=((1, 1), (1, 1))):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(in_planes=self.in_planes, out_planes=planes * BasicBlock.expansion, stride=stride, padding=0),
                norm_layer(num_groups=self.num_groups, num_channels=planes * BasicBlock.expansion))

        layers = [
            BasicBlock(inplanes=self.in_planes, planes=planes, stride=stride, downsample=downsample, padding=padding[0],
                       dilation=self.dilation, num_groups=self.num_groups)]
        self.in_planes = planes * BasicBlock.expansion
        for b in range(1, blocks):
            layers.append(BasicBlock(inplanes=self.in_planes, planes=planes, padding=padding[b], dilation=self.dilation,
                                     num_groups=self.num_groups))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        conv_out = self.layer_4(x)

        x = self.adaptive_pool(conv_out)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return conv_out, x

    def forward(self, x):
        return self._forward_impl(x)
