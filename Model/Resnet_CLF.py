import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.Resnet_2D import Bottleneck,Classifier_Module
affine_par = True#default in batch norm
class Resnet_CLF(nn.Module):
    def __init__(self, _nChannels, _num_classes, _cls_type=None):
        super(Resnet_CLF, self).__init__()

        self.nChannels = _nChannels
        self.in_planes = _nChannels
        self.num_classes = _num_classes
        self.layer=self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],self.num_classes)
        #self.bn1 = nn.BatchNorm2d(self.nChannels, momentum=0.9)
        self.linear = nn.Linear(self.num_classes, self.num_classes)
        self.bn1 = nn.BatchNorm2d(self.num_classes, affine=affine_par)
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(self.nChannels,dilation_series,padding_series,NoLabels)

    def forward(self, feat):
        #print('here is the resnet classifier')
        #print(feat.size())
        feat=self.layer(feat)
        #print(feat.size())
        feat = F.relu(self.bn1(feat))
        feat = F.avg_pool2d(feat, 5)
        feat = feat.view(feat.size(0), -1)
        feat = self.linear(feat)
        return feat