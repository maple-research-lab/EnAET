# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
from Model.TE_Module import TE_Module
import torch
import torch.nn as nn
from Model.Classifier import Classifier
class Connector(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True,
                 indim=256, transform_classes=6,num_class=10,
                nchannels=256*8*8,
                 cls_type='MultLayerFC2',run_type=0):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        #nChannels = 192
        self.run_type=run_type
        super(Connector, self).__init__()
        self.TE = TE_Module(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3,run_type=run_type)
        self.fc = nn.Linear(indim, transform_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.clf=Classifier(_nChannels=nchannels, _num_classes=num_class, _cls_type=cls_type)
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1, x2, out_feat_keys=None):


        if out_feat_keys==None:
            if self.run_type==0:
                use_output_key=['conv2','classifier']
                x1,attention_matrix1 = self.TE(x1, use_output_key)
                x2,attention_matrix2 = self.TE(x2, use_output_key)
                classify_input=x1[0]
                classify_output,_=self.clf(classify_input, False)
                #in this semi-supervised, we do not use attention in the classifier part
                transform_input1=x1[1]
                transform_input2=x2[1]
                x = torch.cat((transform_input1, transform_input2), dim=1)
                transform_output =self.fc(x)
            elif self.run_type==1 or self.run_type==2 or self.run_type==3 or self.run_type==4 or self.run_type==8:
                use_output_key = ['Attention', 'classifier']
                x1, attention_matrix1 = self.TE(x1, use_output_key)
                x2, attention_matrix2 = self.TE(x2, use_output_key)
                classify_input = x1[0]
                classify_output, _ = self.clf(classify_input, False)
                # in this semi-supervised, we do not use attention in the classifier part
                transform_input1 = x1[1]
                transform_input2 = x2[1]
                x = torch.cat((transform_input1, transform_input2), dim=1)
                transform_output = self.fc(x)

        else:
            x1, attention_matrix1 = self.TE(x1, out_feat_keys)
            x2, attention_matrix2 = self.TE(x2, out_feat_keys)
        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)
        if out_feat_keys == None:

            return x1, x2, transform_output,classify_output,attention_matrix1,attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2