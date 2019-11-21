# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
##Pay attention here
#Do not use pre-trained model for weakly supervised learning, that's completely break the law

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
from Model.Attention import Self_Attn
#this is for the 32*32 image input, however, for STL10 input, we need to use another resnet
class PreActResNet_WX(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,run_type=0):
        self.run_type=run_type
        super(PreActResNet_WX, self).__init__()
        self.num_stages = len(num_blocks)
        if run_type == 0:
            blocks = [nn.Sequential() for i in range(self.num_stages)]
        else:
            blocks = [nn.Sequential() for i in range(self.num_stages + 1)]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        count_stage = 0
        blocks[count_stage].add_module('Block1_Conv1', self.conv1)
        blocks[count_stage].add_module('Layer1',self.layer1)
        count_stage+=1
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        blocks[count_stage].add_module('Layer2', self.layer2)
        count_stage += 1
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        blocks[count_stage].add_module('Layer3', self.layer3)
        count_stage += 1
        if run_type==1:
            self.attention = Self_Attn(256, 'relu')
            blocks[count_stage].add_module('Attention', self.attention)
            count_stage += 1

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        blocks[count_stage].add_module('Layer4', self.layer4)
        count_stage += 1
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self._feature_blocks = nn.ModuleList(blocks)
        if run_type == 0:
            self.all_feat_names =['block' + str(s + 1) for s in
                                               range(self.num_stages)] + ['classifier', ]
        elif run_type == 1 or run_type == 2 or run_type == 4 or run_type == 5:
            self.all_feat_names = ['block' + str(s + 1) for s in
                                               range(3)] + ['Attention'] + ['block' + str(s + 1) for s in
                                                                            range(3, self.num_stages)] + [
                                      'classifier', ]
            self.num_stages+=1

    def _parse_out_keys_arg(self, out_feat_keys):
        """
        :param out_feat_keys:
        :return:
        the lasy layer index from out_feat_keys
        """

        # By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1], ] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    'Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {0}.'.format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,out_feat_keys=None):
        go_direct_flag = False
        if out_feat_keys == None:
            go_direct_flag = True
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)

        out_feats = [None] * len(out_feat_keys)
        go_attention_flag = False
        feat = x
        for f in range(max_out_feat + 1):
            key = self.all_feat_names[f]

            if key == 'Attention':
                go_attention_flag = True
                feat, attention = self._feature_blocks[f](feat)
            elif key == 'classifier':

                feat = F.avg_pool2d(feat, 4)
                feat = feat.view(feat.size(0), -1)
                feat = self.linear(feat)
            else:
                feat = self._feature_blocks[f](feat)
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
        if go_direct_flag:
            return out_feats
        # here is an important output link to the classifier
        if go_attention_flag:
            return out_feats, attention
        else:
            return out_feats, None


class PreActResNet_STL(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,run_type=0):
        self.run_type=run_type
        super(PreActResNet_STL, self).__init__()
        self.num_stages = len(num_blocks)
        if run_type == 0:
            blocks = [nn.Sequential() for i in range(self.num_stages)]
        else:
            blocks = [nn.Sequential() for i in range(self.num_stages + 1)]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#first add this to save gpu memory, if it not work, remove this
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        count_stage = 0
        blocks[count_stage].add_module('Block1_Conv1', self.conv1)
        blocks[count_stage].add_module('Block1_BN1', self.bn1)
        blocks[count_stage].add_module('Block1_Relu1', self.relu)
        blocks[count_stage].add_module('Block1_Maxpool1', self.maxpool)
        blocks[count_stage].add_module('Layer1',self.layer1)
        count_stage+=1
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        blocks[count_stage].add_module('Layer2', self.layer2)
        count_stage += 1
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        blocks[count_stage].add_module('Layer3', self.layer3)
        count_stage += 1
        if run_type==1:
            self.attention = Self_Attn(256, 'relu')
            blocks[count_stage].add_module('Attention', self.attention)
            count_stage += 1

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        blocks[count_stage].add_module('Layer4', self.layer4)
        count_stage += 1
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self._feature_blocks = nn.ModuleList(blocks)
        if run_type == 0:
            self.all_feat_names =['block' + str(s + 1) for s in
                                               range(self.num_stages)] + ['classifier', ]
        elif run_type == 1 or run_type == 2 or run_type == 4 or run_type == 5:
            self.all_feat_names = ['block' + str(s + 1) for s in
                                               range(3)] + ['Attention'] + ['block' + str(s + 1) for s in
                                                                            range(3, self.num_stages)] + [
                                      'classifier', ]
            self.num_stages+=1

    def _parse_out_keys_arg(self, out_feat_keys):
        """
        :param out_feat_keys:
        :return:
        the lasy layer index from out_feat_keys
        """

        # By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1], ] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    'Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {0}.'.format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,out_feat_keys=None):
        go_direct_flag = False
        if out_feat_keys == None:
            go_direct_flag = True
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)

        out_feats = [None] * len(out_feat_keys)
        go_attention_flag = False
        feat = x
        for f in range(max_out_feat + 1):
            key = self.all_feat_names[f]

            if key == 'Attention':
                go_attention_flag = True
                feat, attention = self._feature_blocks[f](feat)
            elif key == 'classifier':
                feat = F.avg_pool2d(feat, feat.size(1))
                feat = feat.view(feat.size(0), -1)
                feat = self.linear(feat)
            else:
                feat = self._feature_blocks[f](feat)
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
        if go_direct_flag:
            return out_feats
        # here is an important output link to the classifier
        if go_attention_flag:
            return out_feats, attention
        else:
            return out_feats, None


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34(num_classes,run_type):
    return PreActResNet_WX(PreActBlock, [3,4,6,3],num_classes,run_type)
def PreActResNet34STL(num_classes,run_type):
    return PreActResNet_STL(PreActBlock, [3,4,6,3],num_classes,run_type)


def PreActResNet50(num_classes):
    #Please do not use this
    #The paper "Learning to Learn from Noisy Labeled Data" used this backbone to compare with previous paper
    #Backbone changed and also used pre-trained model to do this. Which is completely unfair
    #Please do not use this to compare with them, they are completely unfair
    return PreActResNet_WX(PreActBottleneck, [3,4,6,3],num_classes)

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152(num_classes,run_type):
    #we plan to use this for cifar10 and cifar100 large model
    return PreActResNet_WX(PreActBlock, [3,8,36,3],num_classes,run_type)