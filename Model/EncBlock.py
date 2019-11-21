# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class EncBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(EncBlock, self).__init__()
        padding =int( (kernel_size -1 ) /2)
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
                                                 kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

    #        self.layers2 = nn.Sequential()
    #        self.layers2.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
    #            kernel_size=kernel_size, stride=1, padding=padding, bias=False))
    #        self.layers2.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = self.layers(x)
        #        out2 = self.layers2(x)
        return torch.cat([x ,out], dim=1)