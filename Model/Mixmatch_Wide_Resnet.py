# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        #print(self.equalInOut)
        #print(x.size())
        #print(out.size())
        #exit()
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
class NetworkBlock_Same(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock_Same, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes, out_planes, stride, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
from Model.Attention import Self_Attn
class WX_WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0,run_type=0,num_stages=4):
        super(WX_WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.num_stages = num_stages
        if run_type == 0:
            blocks = [nn.Sequential() for i in range(self.num_stages)]
        else:
            blocks = [nn.Sequential() for i in range(self.num_stages + 1)]


        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        count_stage=0
        blocks[count_stage].add_module('Block1', self.conv1)
        count_stage += 1
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        blocks[count_stage].add_module('Block2', self.block1)
        count_stage += 1
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        blocks[count_stage].add_module('Block3', self.block2)
        count_stage += 1
        if run_type==1:
            self.attention = Self_Attn(nChannels[2], 'relu')
            blocks[count_stage].add_module('Attention', self.attention)
            count_stage += 1

        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        blocks[count_stage].add_module('Block4', self.block3)
        count_stage += 1
        #add attention in the final layer
        if run_type==2:
            self.attention = Self_Attn(nChannels[3], 'relu')
            blocks[count_stage].add_module('Attention', self.attention)
            count_stage += 1
        self._feature_blocks = nn.ModuleList(blocks)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.run_type=run_type
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        if run_type==0:
            self.all_feat_names = ['block' + str(s + 1) for s in
                                               range(self.num_stages)] + ['classifier', ]
        elif run_type==1:
            self.all_feat_names =  ['block' + str(s + 1) for s in
                                                                       range(3)]+ ['Attention']+['block' + str(s + 1) for s in
                                                                       range(3,self.num_stages)]  + ['classifier', ]

            self.num_stages+=1
        elif run_type==2:
            self.all_feat_names = ['block' + str(s + 1) for s in
                                   range(self.num_stages)]+ ['Attention'] + ['classifier', ]
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
        # out = self.conv1(x)
        # out = self.block1(out)
        # out = self.block2(out)
        # out = self.block3(out)
        # out = self.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        # out = out.view(-1, self.nChannels)
        # return self.fc(out)
        go_direct_flag=False
        if out_feat_keys==None:
            go_direct_flag=True
        elif self.run_type==0 and 'Attention' in out_feat_keys:
            #out_feat_keys.append('block3')
            #out_feat_keys.remove('Attention')
            out_feat_keys[out_feat_keys.index('Attention')] ='block3'
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

                feat = self.relu(self.bn1(feat))
                feat = F.avg_pool2d(feat, 8)
                #feat = feat.view(feat.size(0), -1)
                feat=feat.view(-1, self.nChannels)
                # feat = self.linear(feat)
                feat=self.fc(feat)
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

class WX_WideResNet_STL(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0,run_type=0,num_stages=5):
        super(WX_WideResNet_STL, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor,128*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.num_stages = num_stages
        if run_type == 0:
            blocks = [nn.Sequential() for i in range(self.num_stages)]
        else:
            blocks = [nn.Sequential() for i in range(self.num_stages + 1)]


        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        count_stage=0
        blocks[count_stage].add_module('Block1', self.conv1)
        count_stage += 1
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        blocks[count_stage].add_module('Block2', self.block1)
        count_stage += 1
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        blocks[count_stage].add_module('Block3', self.block2)
        count_stage += 1

        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        blocks[count_stage].add_module('Block4', self.block3)
        count_stage += 1
        #add attention in the final layer
        if run_type==1:
            self.attention = Self_Attn(nChannels[3], 'relu')
            blocks[count_stage].add_module('Attention', self.attention)
            count_stage += 1
        self.block4 = NetworkBlock(n, nChannels[3], nChannels[4], block, 2, dropRate)
        blocks[count_stage].add_module('Block5', self.block4)
        count_stage += 1
        self._feature_blocks = nn.ModuleList(blocks)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[4], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[4], num_classes)
        self.nChannels = nChannels[4]
        self.run_type=run_type
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        if run_type==0:
            self.all_feat_names = ['block' + str(s + 1) for s in
                                               range(self.num_stages)] + ['classifier', ]
        elif run_type==1:
            self.all_feat_names =  ['block' + str(s + 1) for s in
                                                                       range(4)]+ ['Attention']+['block' + str(s + 1) for s in
                                                                       range(4,self.num_stages)]  + ['classifier', ]

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

    def forward(self, x,out_feat_keys=None):
        # out = self.conv1(x)
        # out = self.block1(out)
        # out = self.block2(out)
        # out = self.block3(out)
        # out = self.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        # out = out.view(-1, self.nChannels)
        # return self.fc(out)
        go_direct_flag=False
        if out_feat_keys==None:
            go_direct_flag=True
        elif self.run_type==0 and 'Attention' in out_feat_keys:
            #out_feat_keys.append('block3')
            #out_feat_keys.remove('Attention')
            out_feat_keys[out_feat_keys.index('Attention')] ='block4'
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

                feat = self.relu(self.bn1(feat))
                feat = F.avg_pool2d(feat, 12)#reduce_mean
                #feat = feat.view(feat.size(0), -1)
                feat=feat.view(-1, self.nChannels)
                # feat = self.linear(feat)
                feat=self.fc(feat)
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
#change backbone again because of google, i don't think it makes any sense

class WX_LargeWideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0,run_type=0,num_stages=4):
        super(WX_LargeWideResNet, self).__init__()
        nChannels = [16, 135, 135*widen_factor, 270*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.num_stages = num_stages
        if run_type == 0:
            blocks = [nn.Sequential() for i in range(self.num_stages)]
        else:
            blocks = [nn.Sequential() for i in range(self.num_stages + 1)]


        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        count_stage=0
        blocks[count_stage].add_module('Block1', self.conv1)
        count_stage += 1
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        blocks[count_stage].add_module('Block2', self.block1)
        count_stage += 1
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        blocks[count_stage].add_module('Block3', self.block2)
        count_stage += 1
        if run_type==1:
            self.attention = Self_Attn(nChannels[2], 'relu')
            blocks[count_stage].add_module('Attention', self.attention)
            count_stage += 1

        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        blocks[count_stage].add_module('Block4', self.block3)
        count_stage += 1
        #add attention in the final layer
        if run_type==2:
            self.attention = Self_Attn(nChannels[3], 'relu')
            blocks[count_stage].add_module('Attention', self.attention)
            count_stage += 1
        self._feature_blocks = nn.ModuleList(blocks)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.run_type=run_type
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        if run_type==0:
            self.all_feat_names = ['block' + str(s + 1) for s in
                                               range(self.num_stages)] + ['classifier', ]
        elif run_type==1:
            self.all_feat_names =  ['block' + str(s + 1) for s in
                                                                       range(3)]+ ['Attention']+['block' + str(s + 1) for s in
                                                                       range(3,self.num_stages)]  + ['classifier', ]

            self.num_stages+=1
        elif run_type==2:
            self.all_feat_names = ['block' + str(s + 1) for s in
                                   range(self.num_stages)]+ ['Attention'] + ['classifier', ]
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
        # out = self.conv1(x)
        # out = self.block1(out)
        # out = self.block2(out)
        # out = self.block3(out)
        # out = self.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        # out = out.view(-1, self.nChannels)
        # return self.fc(out)
        go_direct_flag=False
        if out_feat_keys==None:
            go_direct_flag=True
        elif self.run_type==0 and 'Attention' in out_feat_keys:
            #out_feat_keys.append('block3')
            #out_feat_keys.remove('Attention')
            out_feat_keys[out_feat_keys.index('Attention')] ='block3'
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

                feat = self.relu(self.bn1(feat))
                feat = F.avg_pool2d(feat, 8)
                #feat = feat.view(feat.size(0), -1)
                feat=feat.view(-1, self.nChannels)
                # feat = self.linear(feat)
                feat=self.fc(feat)
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
