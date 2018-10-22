# coding=utf-8
'''
https://github.com/pytorch/vision.git
'''

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #add by Leon,origional is as follows
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc = nn.Linear(512 * 64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print("x0",x.size())
        x = self.conv1(x)
        #print("x1",x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print("x2", x.size())
        x = self.layer1(x)
        #print("x3", x.size())
        x = self.layer2(x)
        #print("x4", x.size())
        x = self.layer3(x)
        #print("x5", x.size())
        x = self.layer4(x)
        #print("x6", x.size())
        x = self.avgpool(x)
        #print("x7", x.size())
        x = x.view(x.size(0), -1)
        #print("x8" ,x.size())
        x = self.fc(x)
        #print("x9" ,x)


        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_classes = kwargs.get('num_classes', 1000)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_classes = kwargs.get('num_classes', 1000)
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_classes = kwargs.get('num_classes', 1000)
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_classes = kwargs.get('num_classes', 1000)
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_classes = kwargs.get('num_classes', 1000)
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model


def re_define_layers(model, **kwargs):
    """
    for fin
    :param kwargs: 
    :return: 
    """
    model_net_name = kwargs.get('model_net_name','resnet101')
    num_classes = kwargs.get('num_classes', 1000)
    if model_net_name == 'resnet18':
        block = BasicBlock
    elif model_net_name == 'resnet34':
        block = BasicBlock
    elif model_net_name == 'resnet50':
        block = Bottleneck
    elif model_net_name == 'resnet101':
        block = Bottleneck
    elif model_net_name == 'resnet152':
        block = Bottleneck
    else:
        raise ValueError('model_net_name error !')
    model.fc = nn.Linear(512 * block.expansion, num_classes)


if __name__ == "__main__":
    print('begin...')
    print('done')