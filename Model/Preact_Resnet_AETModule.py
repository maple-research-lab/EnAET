# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Preact_Resnet import PreActBottleneck,PreActBlock

class Preact_Resnet_AETModule(nn.Module):
    def __init__(self, num_layer_in_block=3,num_classes=8,channels=512):
        super(Preact_Resnet_AETModule, self).__init__()
        self.channels=channels
        self.in_planes=int(channels/2)
        self.layer4 = self._make_layer(PreActBlock, self.channels, num_layer_in_block, stride=2)
        self.linear = nn.Linear(self.channels * PreActBlock.expansion*2, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x1, x2):
        #if self.run_type==0 or self.run_type==2:#need block or not
        x1=self.block(x1)
        x2=self.block(x2)
        x = torch.cat((x1, x2), dim=1)
        return self.linear(x)
    def block(self,x):
        out = self.layer4(x)
        out = F.avg_pool2d(out, out.size(1))
        out = out.view(out.size(0), -1)
        return out