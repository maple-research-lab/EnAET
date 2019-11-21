# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
#from Sime_Supervised.TE_Module import TE_Module
import torch
import torch.nn as nn
from Model.Wide_Resnet import My_Wide_ResNet
from Model.Resnet_Classifier import Classifier_Resnet
class Connector_Resnet(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True,
                 depth=28,width=2,dropout_rate=0.3,
                  transform_classes=6,num_class=10,
                nchannels=64,aet_channels=128,
                 cls_type='MultLayerFC2',run_type=0):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        #nChannels = 192
        self.run_type=run_type
        super(Connector_Resnet, self).__init__()
        #self.TE = TE_Module(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3,run_type=run_type)
        self.fc = nn.Linear(aet_channels*2, transform_classes)
        self.wide_resnet=My_Wide_ResNet(depth, width, dropout_rate, transform_classes,num_stages=_num_stages,run_type=run_type)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.clf=Classifier_Resnet(_nChannels=nchannels, _num_classes=num_class, _cls_type=cls_type)
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1, x2, out_feat_keys=None):


        if out_feat_keys==None:
            if self.run_type==0:
                use_output_key=['block2','classifier']
                x1,attention_matrix1 = self.wide_resnet(x1, use_output_key)
                x2,attention_matrix2 = self.wide_resnet(x2, use_output_key)
                classify_input=x1[0]
                #print(classify_input)
                classify_output=self.clf(classify_input)
                #in this semi-supervised, we do not use attention in the classifier part
                transform_input1=x1[1]
                transform_input2=x2[1]
                x = torch.cat((transform_input1, transform_input2), dim=1)
                transform_output = self.fc(x)
            elif self.run_type==1 or self.run_type==2 or self.run_type==4 or self.run_type==5:
                #print('call connector forward')
                use_output_key = ['Attention', 'classifier']
                x1, attention_matrix1 = self.wide_resnet(x1, use_output_key)
                x2, attention_matrix2 = self.wide_resnet(x2, use_output_key)

                # in this semi-supervised, we do not use attention in the classifier part
                transform_input1 = x1[1]
                transform_input2 = x2[1]
                #print(transform_input1.size())
                x = torch.cat((transform_input1, transform_input2), dim=1)
                transform_output = self.fc(x)
                classify_input = x1[0]
                #print(classify_input.size())
                classify_output = self.clf(classify_input)

        else:
            x1, attention_matrix1 = self.wide_resnet(x1, out_feat_keys)
            x2, attention_matrix2 = self.wide_resnet(x2, out_feat_keys)
        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)
        if out_feat_keys == None:
            return x1, x2, transform_output,classify_output,attention_matrix1,attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2