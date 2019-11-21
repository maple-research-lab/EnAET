# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.TE_Module import TE_Module
class Classifier_Mixmatch(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True, indim=128, num_classes=10,run_type=0):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        #nChannels = 192
        super(Classifier_Mixmatch, self).__init__()
        self.clf = TE_Module(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3,run_type=run_type)
        self.fc = nn.Linear(indim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1,out_feat_keys=None):

        x1,atten = self.clf(x1,out_feat_keys)
        if out_feat_keys==None:
            return self.fc(x1)
        else:
            return x1,atten
