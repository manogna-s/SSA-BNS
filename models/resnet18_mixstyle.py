from .resnet18 import *
from .mixstyle import MixStyle



class ResNet_Mixstyle(nn.Module):

    def __init__(self, block, layers, classifier=None, num_classes=64,
                 dropout=0.0, global_pool=True):
        super(ResNet_Mixstyle, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.outplanes = 512
        self.mixstyle = MixStyle(p=0.5, alpha=0.1, mix='random')

        # handle classifier creation
        if num_classes is not None:
            if classifier == 'linear':
                self.cls_fn = nn.Linear(self.outplanes, num_classes)
            elif classifier == 'cosine':
                self.cls_fn = CosineClassifier(self.outplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        x = self.cls_fn(x)
        return x

    def embed(self, x, param_dict=None):
        x = self.conv1(x)
        x = self.bn1(x)
        # print(self.bn1.training, self.bn1.track_running_stats)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.mixstyle(x)
        x = self.layer2(x)
        x = self.mixstyle(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.squeeze()

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]


def resnet18_mixstyle(pretrained=False, pretrained_model_path=None, **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    model = ResNet_Mixstyle(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        device = model.get_state_dict()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_parameters(ckpt_dict['state_dict'], strict=False)
        print('Loaded shared weights from {}'.format(pretrained_model_path))
    return model
