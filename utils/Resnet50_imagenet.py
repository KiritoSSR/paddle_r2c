import paddle
import paddle.nn as nn

import pickle
import numpy as np


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, dilation=dilation)


def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride)


class BottleneckBlock(nn.Layer):

    expansion = 4
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 num_layers=1):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups
        if num_layers == 1 or num_layers == 4:
            self.conv1 = conv1x1(inplanes, width, stride = 1)
        else:
            self.conv1 = conv1x1(inplanes, width, stride = 2)
        self.bn1 = norm_layer(width)
        if num_layers == 1:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, 1, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        if num_layers == 4:
            self.downsample = nn.Sequential(
                conv1x1(1024, planes *4, stride=1),
                norm_layer(planes * 4),
            )
        else:
            self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Layer):
    def __init__(
        self,
        block,
        layers,
        num_classes = 1000,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],num_layers=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],num_layers=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],num_layers=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],num_layers=4)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks,
                    stride = 1, dilate = False,num_layers=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,num_layers))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers,**kwargs):
    model = ResNet(block, layers,**kwargs)
    return model

def resnet50_imagenet(**kwargs):
    return _resnet('resnet50', BottleneckBlock, [3, 4, 6, 3])

# if __name__ == "__main__":
#     dummy_input = [paddle.ones(shape=[3, 224, 224])]
#     model = resnet50()
#     print(model)
#     print(model.state_dict().keys())
#     with open('torch_resnet50.pkl', 'rb') as f:
#         param2 = pickle.load(f)
#     print(len(param2.keys()))
#     model.set_state_dict(param2)
#     model.eval()
#     #
#     output = model(paddle.to_tensor(dummy_input))
#     np.save('paddle_resnet50.npy', output.numpy())