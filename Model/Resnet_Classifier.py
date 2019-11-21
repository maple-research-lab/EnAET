import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.Wide_Resnet import wide_basic
class Classifier_Resnet(nn.Module):
    def __init__(self, _nChannels, _num_classes, _cls_type=None,dropout_rate=0.3,blocks=4):
        super(Classifier_Resnet, self).__init__()

        self.nChannels = _nChannels*2
        self.in_planes = _nChannels
        self.num_classes = _num_classes
        self.layer3 = self._wide_layer(wide_basic, self.nChannels, blocks,dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(self.nChannels, momentum=0.9)
        self.linear = nn.Linear(self.nChannels, self.num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    def forward(self, feat):
        #print('here is the resnet classifier')
        #print(feat.size())
        feat=self.layer3(feat)
        feat = F.relu(self.bn1(feat))
        feat = F.avg_pool2d(feat, 8)
        feat = feat.view(feat.size(0), -1)
        feat = self.linear(feat)
        return feat