import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'resnet18': ('https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
                 'cf548f46534aa3560945be4b95cd11c4'),
    'resnet34': ('https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
                 '8d2275cf8706028345f78ac0e1d31969'),
    'resnet50': ('https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
                 'ca6f485ee1ab0492d38f323885b0ad80'),
    'resnet101': ('https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
                  '02f35f034ca3858e1e54d4036443c92d'),
    'resnet152': ('https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams',
                  '7ad16a2f1e7333859ff986138630fd7a'),
}

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
                 num_layers = 1):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        if num_layers == 1 or num_layers == 4 :
            self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        else:
            self.conv1 = nn.Conv2D(inplanes, width, 1,stride=2, bias_attr=False)
        self.bn1 = norm_layer(width)
        if num_layers == 1:
            self.conv2 = nn.Conv2D(
                    width,
                    width,
                    3,
                    padding=dilation,
                    stride=stride,
                    groups=groups,
                    dilation=dilation,
                    bias_attr=False)
        else:
            self.conv2 = nn.Conv2D(
                    width,
                    width,
                    3,
                    padding=dilation,
                    stride=1,
                    groups=groups,
                    dilation=dilation,
                    bias_attr=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2D(
            width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
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
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.
    Examples:
        .. code-block:: python
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock
            resnet50 = ResNet(BottleneckBlock, 50)
            resnet18 = ResNet(BasicBlock, 18)
    """

    def __init__(self, block, depth, num_classes=1000, with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],num_layers=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,num_layers=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,num_layers=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,num_layers=4)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,num_layers=1 ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []


        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64,
                  previous_dilation, norm_layer,num_layers))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """ResNet 50-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Examples:
        .. code-block:: python
            from paddle.vision.models import resnet50
            # build model
            model = resnet50()
            # build model and load imagenet pretrained weight
            # model = resnet50(pretrained=True)
    """
    return _resnet('resnet50', BottleneckBlock, 50, pretrained, **kwargs)





def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model




def resnet50(pretrained=False, **kwargs):
    """ResNet 50-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Examples:
        .. code-block:: python
            from paddle.vision.models import resnet50
            # build model
            model = resnet50()
            # build model and load imagenet pretrained weight
            # model = resnet50(pretrained=True)
    """
    return _resnet('resnet50', BottleneckBlock, 50, pretrained, **kwargs)





