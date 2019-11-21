# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch
import torch.nn as nn
from Model.Resnet_CLF import Resnet_CLF
from Model.Resnet_2D import Atten_ResNet,Bottleneck

class Connector_Resnet2D(nn.Module):
    def __init__(self,
                  transform_classes=6,num_class=10,
                nchannels=512,aet_channels=128,
                 cls_type='MultLayerFC2',run_type=0):
        """
                :param _num_stages: block combination
                :param _use_avg_on_conv3: finally use avg or not
                :param indim:
                :param num_classes: transformation matrix
                """
        super(Connector_Resnet2D, self).__init__()
        self.fc = nn.Linear(aet_channels * 2, transform_classes)
        self.Backbone=Atten_ResNet(Bottleneck, [3, 8, 36, 3],aet_channels,num_stages=6)
        self.clf=Resnet_CLF(nchannels, num_class, _cls_type=cls_type)
        self.run_type=run_type
    def forward(self, x1, x2, out_feat_keys=None):

        if out_feat_keys==None:
            if self.run_type==0:
                use_output_key=['block2','classifier']
                x1,attention_matrix1 = self.Backbone(x1, use_output_key)
                x2,attention_matrix2 = self.Backbone(x2, use_output_key)
                classify_input=x1[0]
                #print(classify_input)
                classify_output=self.clf(classify_input)
                #in this semi-supervised, we do not use attention in the classifier part
                transform_input1=x1[1]
                transform_input2=x2[1]
                x = torch.cat((transform_input1, transform_input2), dim=1)
                transform_output =self.fc(x)


        else:
            x1, attention_matrix1 = self.Backbone(x1, out_feat_keys)
            x2, attention_matrix2 = self.Backbone(x2, out_feat_keys)
        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)

        if out_feat_keys == None:
            return x1, x2, transform_output,classify_output,attention_matrix1,attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2

