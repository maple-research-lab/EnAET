# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    #a block combined with bn>relu>conv
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
from Model.Attention import Self_Attn
class My_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes,num_stages=3,run_type=0):
        super(My_Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.num_stages=num_stages
        if run_type==0:
            blocks = [nn.Sequential() for i in range(len(nStages))]
        else:
            blocks = [nn.Sequential() for i in range(len(nStages)+1)]
        self.conv1 = conv3x3(3,nStages[0])
        count_stage=0
        blocks[count_stage].add_module('Block1_Conv1',self.conv1)
        count_stage+=1
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        blocks[count_stage].add_module('Block2_layer1', self.layer1)
        count_stage += 1
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        blocks[count_stage].add_module('Block3_layer2', self.layer2)
        count_stage += 1
        if run_type==1 or run_type==2 or run_type==4 or run_type==5:
            self.attention = Self_Attn(nStages[2], 'relu')
            blocks[count_stage].add_module('Attention', self.attention)
            count_stage += 1
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        blocks[count_stage].add_module('Block4_layer3', self.layer3)
        count_stage += 1
        self._feature_blocks = nn.ModuleList(blocks)

        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        self.run_type=run_type
        if run_type==0:
            self.all_feat_names = ['conv1'] + ['block' + str(s + 1) for s in
                                               range(self.num_stages)] + ['classifier', ]
        elif run_type==1 or run_type==2 or run_type==4 or run_type==5:
            self.all_feat_names = ['conv1'] + ['block' + str(s + 1) for s in
                                                                       range(2)]+ ['Attention']+['block' + str(s + 1) for s in
                                                                       range(2,self.num_stages)]  + ['classifier', ]

            self.num_stages+=1
        #out_feat_keys= ['Attention']
        #out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        #print(max_out_feat)
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        #print(num_blocks)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

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
    def forward(self, x,out_feat_keys=None):
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)
        go_attention_flag = False
        feat = x
        for f in range(max_out_feat + 1):
            key = self.all_feat_names[f]

            if key == 'Attention':
                go_attention_flag = True
                feat, attention = self._feature_blocks[f](feat)
            elif key=='classifier':

                feat = F.relu(self.bn1(feat))
                feat = F.avg_pool2d(feat, 8)
                feat = feat.view(feat.size(0), -1)
                #feat = self.linear(feat)
            else:
                feat = self._feature_blocks[f](feat)
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats

        #here is an important output link to the classifier
        if go_attention_flag:
            return out_feats, attention
        else:
            return out_feats, None