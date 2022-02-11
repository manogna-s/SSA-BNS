from .resnet18 import *
import random
import numpy as np


def deactivate_imix(m):
    # print('Dectivating imix')
    if type(m) == IMix:
        m.set_activation_status(False)

def activate_imix(m):
    # print('Activating imix')
    if type(m) == IMix:
        m.set_activation_status(True)


class IMix(nn.Module):

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using iMix.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.lamda= 0.8
        print(f'Using {mix} IMIX with lambda {self.lamda}')

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status


    def forward(self, x):
        # print(self.training, self._activated)
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = torch.maximum(lmda, 1-lmda)
        if self.lamda != None:
            lmda[:] = self.lamda
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        return lmda*x + (1-lmda)*x[perm]

    def tsne_feats(self, support_features): #, query_features, query_labels, num_classes):

        B = support_features.shape[0]
        
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = torch.maximum(lmda, 1-lmda)
        if self.lamda != None:
            lmda[:] = self.lamda
        lmda = lmda.to(support_features.device)
        perm = torch.randperm(B)
        perturbed_features = lmda*support_features + (1-lmda)*support_features[perm]

        return perturbed_features




class ResNet_iMix(nn.Module):

    def __init__(self, block, layers, classifier=None, num_classes=64,
                 dropout=0.0, global_pool=True):
        super(ResNet_iMix, self).__init__()
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
        self.num_classes = num_classes
        self.imix = IMix(p=0.5, alpha=0.1, mix='random')

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
        x = self.imix(x)
        x = self.layer2(x)
        # x = self.imix(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.squeeze()

    def get_layer1_features(self, episode):
        support_images, support_labels = episode['support_images'], episode['support_labels']
        query_images, query_labels = episode['query_images'], episode['query_labels']

        #support features
        x= support_images
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)
        support_features = self.layer1(x)

        #query features
        x= query_images
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)
        query_features = self.layer1(x)


        features, labels = self.imix.tsne_feats(support_features, support_labels, query_features, query_labels, self.num_classes)
        features = features.cpu().data.numpy()
        return features, labels
        
    def get_final_features(self, episode):
        support_images, support_labels = episode['support_images'], episode['support_labels']
        query_images, query_labels = episode['query_images'], episode['query_labels']

        #support features
        x= support_images
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)
        support_features = self.layer1(x)
        perturbed_features = self.imix.tsne_feats(support_features)

        x = self.layer2(support_features)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        support_features = x.squeeze()

        x = self.layer2(perturbed_features)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        perturbed_features = x.squeeze()


        #query features
        query_features = self.embed(query_images)


        features = [support_features, 
                    perturbed_features, 
                    query_features,  
                    compute_prototypes(support_features, support_labels, self.num_classes)]
        
        support_labels = support_labels.cpu().data.numpy()
        query_labels = query_labels.cpu().data.numpy()
        labels =   [support_labels, 
                    np.ones_like(support_labels)*self.num_classes + support_labels, 
                    np.ones_like(query_labels)*2*self.num_classes + query_labels, 
                    4*self.num_classes+np.arange(self.num_classes)]

        if self.cls_fn!=None:
            features.append(self.cls_fn.weight.T)
            labels.append(3*self.num_classes+np.arange(self.num_classes))

        
        labels = np.concatenate(labels)
        features = torch.vstack(features)
        features = torch.nn.functional.normalize(features, p=2, dim=-1, eps=1e-12)
        features = features.cpu().data.numpy()

        return features, labels

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]


def resnet18_imix(pretrained=False, pretrained_model_path=None, **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    model = ResNet_iMix(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        device = model.get_state_dict()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_parameters(ckpt_dict['state_dict'], strict=False)
        print('Loaded shared weights from {}'.format(pretrained_model_path))
    return model
