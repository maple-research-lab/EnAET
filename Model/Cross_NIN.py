# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import math
import torch.nn as nn
from Model.Basic_Block import BasicBlock,GlobalAveragePooling
from Model.Cross_Attention import Cross_Attn2
from Model.Attention import Self_Attn
class Cross_NIN(nn.Module):
    def __init__(self, _num_inchannels=3, _num_stages=3, _use_avg_on_conv3=True):
        super(Cross_NIN, self).__init__()

        num_inchannels = _num_inchannels
        num_stages = _num_stages
        use_avg_on_conv3 = _use_avg_on_conv3

        assert (num_stages >= 3)
        nChannels = 192
        nChannels2 = 160
        nChannels3 = 96

        blocks = [nn.Sequential() for i in range(num_stages+2)]
        # 1st block,kernel size 5 to 1 to 1
        blocks[0].add_module('Block1_ConvB1', BasicBlock(num_inchannels, nChannels, 5))
        blocks[0].add_module('Block1_ConvB2', BasicBlock(nChannels, nChannels2, 1))
        blocks[0].add_module('Block1_ConvB3', BasicBlock(nChannels2, nChannels3, 1))
        blocks[0].add_module('Block1_MaxPool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # 2nd block
        blocks[1].add_module('Block2_ConvB1', BasicBlock(nChannels3, nChannels, 5))
        blocks[1].add_module('Block2_ConvB2', BasicBlock(nChannels, nChannels, 1))
        blocks[1].add_module('Block2_ConvB3', BasicBlock(nChannels, nChannels, 1))
        blocks[1].add_module('Block2_AvgPool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        #blocks.append(nn.Sequential())

        blocks[2].add_module('Attention', Self_Attn(nChannels, 'relu'))
        blocks[3].add_module('Cross', Cross_Attn2(nChannels, 'relu'))
        # 3rd block
        blocks[4].add_module('Block3_ConvB1', BasicBlock(nChannels, nChannels, 3))
        blocks[4].add_module('Block3_ConvB2', BasicBlock(nChannels, nChannels, 1))
        blocks[4].add_module('Block3_ConvB3', BasicBlock(nChannels, nChannels, 1))

        if num_stages > 5 and use_avg_on_conv3:
            blocks[4].add_module('Block3_AvgPool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        for s in range(5, num_stages+2):
            blocks[s].add_module('Block' + str(s + 1) + '_ConvB1', BasicBlock(nChannels, nChannels, 3))
            blocks[s].add_module('Block' + str(s + 1) + '_ConvB2', BasicBlock(nChannels, nChannels, 1))
            blocks[s].add_module('Block' + str(s + 1) + '_ConvB3', BasicBlock(nChannels, nChannels, 1))
        #attention module

        # global average pooling and classifier
        blocks.append(nn.Sequential())
        blocks[-1].add_module('GlobalAveragePooling', GlobalAveragePooling())
        # for i,block in enumerate(blocks):
        #     print('-'*10+'Module'+str(i)+'-'*10)
        #     for module in block.modules():
        #         print(module)
        #self.final_pool=GlobalAveragePooling()
        #self.attention_module=Self_Attn(nChannels,'relu')
        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ['conv1']+['conv2']+['Attention']+['Cross']+['conv' + str(s + 1) for s in range(2,num_stages)] + ['classifier', ]
        #example ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'classifier']
        assert (len(self.all_feat_names) == len(self._feature_blocks))
        self.weight_initialization()

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

    def forward(self, x, input_attention=None,out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.
        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.
        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        if max_out_feat<=2:
            first_part=True
            assert input_attention==None
        else:
            first_part=False
            #assert input_attention != None
        out_feats = [None] * len(out_feat_keys)
        go_attention_flag=False
        if first_part:
            feat = x
            for f in range(max_out_feat + 1):
                key = self.all_feat_names[f]
                if key=='Attention':
                    go_attention_flag=True
                    feat,attention=self._feature_blocks[f](feat)
                else:
                    feat = self._feature_blocks[f](feat)
                if key in out_feat_keys:
                    out_feats[out_feat_keys.index(key)] = feat
        else:
            feat = x
            for f in range(3,max_out_feat + 1):
                key = self.all_feat_names[f]
                if key == 'Cross':
                    #assert f==3
                    feat= self._feature_blocks[f]((feat,input_attention))
                else:
                    feat = self._feature_blocks[f](feat)
                if key in out_feat_keys:
                    out_feats[out_feat_keys.index(key)] = feat
        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
        if go_attention_flag:
            return out_feats,attention
        else:
            return out_feats,None

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):#init with not transform on batchnorm
                if m.weight.requires_grad:
                    m.weight.data.fill_(1)
                if m.bias.requires_grad:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias.requires_grad:
                    m.bias.data.zero_()