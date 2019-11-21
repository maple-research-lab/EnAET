# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from spectral import SpectralNorm
import numpy as np


class Cross_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Cross_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation


        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x,attention):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
class Cross_Attn2(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Cross_Attn2, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation


        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x,attention=x
        m_batchsize, C, width, height = x.size()
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out