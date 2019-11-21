# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import math
import torch.nn as nn
import numpy as np
from Model.Basic_Block import Flatten,BasicBlock,GlobalAveragePooling
from Model.Attention import Self_Attn
from Model.TEBlock import TEBlock1
class Classifier(nn.Module):
    def __init__(self, _nChannels, _num_classes, _cls_type):
        super(Classifier, self).__init__()
        nChannels = _nChannels
        num_classes = _num_classes
        self.cls_type = _cls_type

        self.classifier = nn.Sequential()
        #first add a self attention part
        self.attention_module=Self_Attn(int(nChannels/8/8),'relu')

        if self.cls_type == 'MultLayer':
            nFeats = min(num_classes * 20, 2048)
            self.classifier.add_module('Flatten', Flatten())
            self.classifier.add_module('Liniear_1', nn.Linear(nChannels, nFeats, bias=False))
            self.classifier.add_module('BatchNorm_1', nn.BatchNorm1d(nFeats))
            self.classifier.add_module('ReLU_1', nn.ReLU(inplace=True))
            self.classifier.add_module('Liniear_2', nn.Linear(nFeats, nFeats, bias=False))
            self.classifier.add_module('BatchNorm2d', nn.BatchNorm1d(nFeats))
            self.classifier.add_module('ReLU_2', nn.ReLU(inplace=True))
            self.classifier.add_module('Liniear_F', nn.Linear(nFeats, num_classes))

        elif self.cls_type == 'MultLayerFC1':
            self.classifier.add_module('Batchnorm', nn.BatchNorm2d(nChannels / 8 / 8, affine=False))
            self.classifier.add_module('Flatten', Flatten())
            self.classifier.add_module('Liniear_F', nn.Linear(nChannels, num_classes))
        elif self.cls_type == 'MultLayerFC2':
            nFeats = min(num_classes * 20, 2048)
            self.classifier.add_module('Flatten', Flatten())
            self.classifier.add_module('Liniear_1', nn.Linear(nChannels, nFeats, bias=False))
            self.classifier.add_module('BatchNorm_1', nn.BatchNorm1d(nFeats))
            self.classifier.add_module('ReLU_1', nn.ReLU(inplace=True))
            self.classifier.add_module('Liniear_F', nn.Linear(nFeats, num_classes))

        elif self.cls_type == 'NIN_ConvBlock3':
            self.classifier.add_module('Block3_ConvB1', BasicBlock(nChannels, 192, 3))
            self.classifier.add_module('Block3_ConvB2', BasicBlock(192, 192, 1))
            self.classifier.add_module('Block3_ConvB3', BasicBlock(192, 192, 1))
            self.classifier.add_module('GlobalAvgPool', GlobalAveragePooling())
            self.classifier.add_module('Liniear_F', nn.Linear(192, num_classes))
        elif self.cls_type == 'Alexnet_conv5' or self.cls_type == 'Alexnet_conv4':
            if self.cls_type == 'Alexnet_conv4':
                block5 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                self.classifier.add_module('ConvB5', block5)
            self.classifier.add_module('Pool5', nn.MaxPool2d(kernel_size=3, stride=2))
            self.classifier.add_module('Flatten', Flatten())
            self.classifier.add_module('Linear1', nn.Linear(256 * 6 * 6, 4096, bias=False))
            self.classifier.add_module('BatchNorm1', nn.BatchNorm1d(4096))
            self.classifier.add_module('ReLU1', nn.ReLU(inplace=True))
            self.classifier.add_module('Liniear2', nn.Linear(4096, 4096, bias=False))
            self.classifier.add_module('BatchNorm2', nn.BatchNorm1d(4096))
            self.classifier.add_module('ReLU2', nn.ReLU(inplace=True))
            self.classifier.add_module('LinearF', nn.Linear(4096, num_classes))
        elif self.cls_type == 'l3layer_conv3':
            nChannels = 128
            nChannels1 = 256
            nChannels2 = 512
            self.classifier.add_module('Block3_ConvB1', TEBlock1(nChannels1, nChannels2, 3))#no padding
            self.classifier.add_module('Block3_ConvB2', TEBlock1(nChannels2, nChannels1, 1))
            self.classifier.add_module('Block3_ConvB3', TEBlock1(nChannels1, nChannels, 1))
            self.classifier.add_module('GlobalAvgPool', GlobalAveragePooling())
            self.classifier.add_module('Linear_F', nn.Linear(128, num_classes))
        else:
            raise ValueError('Not recognized classifier type: %s' % self.cls_type)

        self.initilize()

    def forward(self, feat,label):
        attention=None
        if label:
            feat,attention=self.attention_module(feat)
        feat=self.classifier(feat)
        return feat,attention
    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0 / fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
