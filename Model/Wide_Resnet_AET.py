# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch
import torch.nn as nn
from Model.Mixmatch_Wide_Resnet import NetworkBlock,BasicBlock,NetworkBlock_Same
import torch.nn.functional as F
class Wide_Resnet_AET_Model(nn.Module):
    def __init__(self, num_layer_in_block=4,num_classes=8,dropRate=0.3,widen_factor=2,run_type=0):
        super(Wide_Resnet_AET_Model, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor,128*widen_factor]
        if run_type==0:
            self.block=NetworkBlock(num_layer_in_block, nChannels[2], nChannels[3], BasicBlock, 2, dropRate)
        elif run_type==1:
            self.block=NetworkBlock_Same(num_layer_in_block, nChannels[3], nChannels[3], BasicBlock, 1, dropRate)
        elif run_type==2:
            self.block=NetworkBlock(num_layer_in_block, nChannels[3], nChannels[4], BasicBlock, 2, dropRate)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if run_type!=2:
            self.nChannels = nChannels[3]
            self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        else:
            self.nChannels=nChannels[4]
            self.bn1 = nn.BatchNorm2d(nChannels[4], momentum=0.001)
        self.fc = nn.Linear(self.nChannels * 2, num_classes)
        self.run_type=run_type
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x1, x2):
        #if self.run_type==0 or self.run_type==2:#need block or not
        x1=self.block(x1)
        x2=self.block(x2)
        x1=self.call_block(x1)
        x2=self.call_block(x2)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)
    def call_block(self,feat):
        feat = self.relu(self.bn1(feat))
        if self.run_type==2:
            feat = F.avg_pool2d(feat, 12)
        else:
            feat = F.avg_pool2d(feat, 8)#now the
        # feat = feat.view(feat.size(0), -1)
        feat = feat.view(-1, self.nChannels)
        return feat


class Wide_Resnet_AET_LargeModel(nn.Module):
    def __init__(self, num_layer_in_block=4,num_classes=8,dropRate=0.3,widen_factor=2):
        super(Wide_Resnet_AET_LargeModel, self).__init__()
        nChannels = [16, 135, 135*widen_factor, 270*widen_factor]
        self.block=NetworkBlock(num_layer_in_block, nChannels[2], nChannels[3], BasicBlock, 2, dropRate)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.nChannels=nChannels[3]
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.fc = nn.Linear(self.nChannels * 2, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x1, x2):
        #if self.run_type==0 or self.run_type==2:#need block or not
        x1=self.block(x1)
        x2=self.block(x2)
        x1=self.call_block(x1)
        x2=self.call_block(x2)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)
    def call_block(self,feat):
        feat = self.relu(self.bn1(feat))
        feat = F.avg_pool2d(feat, 8)
        feat = feat.view(-1, self.nChannels)
        return feat
#do not work if share parameters for aet part
class Wide_Resnet_AET_Model_Share(nn.Module):
    def __init__(self, num_layer_in_block=4,dropRate=0.3,widen_factor=2,run_type=0):
        super(Wide_Resnet_AET_Model_Share, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor,128*widen_factor]
        if run_type==0:
            self.block=NetworkBlock(num_layer_in_block, nChannels[2], nChannels[3], BasicBlock, 2, dropRate)
        elif run_type==1:
            self.block=NetworkBlock_Same(num_layer_in_block, nChannels[3], nChannels[3], BasicBlock, 1, dropRate)
        elif run_type==2:
            self.block=NetworkBlock(num_layer_in_block, nChannels[3], nChannels[4], BasicBlock, 2, dropRate)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if run_type!=2:
            self.nChannels = nChannels[3]
            self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        else:
            self.nChannels=nChannels[4]
            self.bn1 = nn.BatchNorm2d(nChannels[4], momentum=0.001)

        self.run_type=run_type

    def forward(self, x):
        #if self.run_type==0 or self.run_type==2:#need block or not
        x1=self.block(x)
        x1=self.call_block(x1)
        return x1
    def call_block(self,feat):
        feat = self.relu(self.bn1(feat))
        #if self.run_type!=2:
        feat = F.avg_pool2d(feat, 8)
        #else:
        #    feat = F.avg_pool2d(feat, 4)#now the
        # feat = feat.view(feat.size(0), -1)
        feat = feat.view(-1, self.nChannels)
        return feat
