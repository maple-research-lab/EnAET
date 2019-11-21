import torch
import torch.nn as nn
from Model.Classifier import Classifier
from Model.Cross_TE_Module import Cross_TE_Module

class TE_Cross_Connector(nn.Module):
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
        self.run_type = run_type
        super(TE_Cross_Connector, self).__init__()
        self.Cross_TE = Cross_TE_Module(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3, run_type=run_type)
        self.fc = nn.Linear(indim, transform_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.clf = Classifier(_nChannels=nchannels, _num_classes=num_class, _cls_type=cls_type)
    def forward(self, x1, x2, out_feat_keys=None):
        first_stage_list = ['conv1'] + ['conv2'] + ['Attention']
        check_flag = True
        if out_feat_keys!=None:
            for item in out_feat_keys:
                if item not in first_stage_list:
                    check_flag = False
        else:
            check_flag=False
        if check_flag:
            # only use first stage: cross-attention trained result directly applied to itself
            x1, attention_matrix1 = self.Cross_TE(x1, out_feat_keys=out_feat_keys)
            x2, attention_matrix2 = self.Cross_TE(x2, out_feat_keys=out_feat_keys)

        elif out_feat_keys==None:
            out_feat_key1 = 'Attention'
            specific_key = 'conv2'

            x1, attention_matrix1 = self.Cross_TE(x1, out_feat_keys=[specific_key, out_feat_key1])
            x2, attention_matrix2 = self.Cross_TE(x2, out_feat_keys=[specific_key, out_feat_key1])
            feat1 = x1[0]  # output from conv without attention applying
            feat2 = x2[0]
            feat_clf1=x1[1]
            new_key=['Cross','classifier']
            x1, cross_matrix1 = self.Cross_TE(feat1, input_attention=attention_matrix2,out_feat_keys=new_key)
            x2, cross_matrix2 = self.Cross_TE(feat2, input_attention=attention_matrix1,out_feat_keys=new_key)

            classify_input = feat_clf1
            classify_output, _ = self.clf(classify_input, False)
            # in this semi-supervised, we do not use attention in the classifier part
            transform_input1 = x1[1]
            transform_input2 = x2[1]
            x = torch.cat((transform_input1, transform_input2), dim=1)
            transform_output = torch.tanh(self.fc(x))

        else:
            x1, attention_matrix1 = self.TE(x1, out_feat_keys)
            x2, attention_matrix2 = self.TE(x2, out_feat_keys)
        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)
        if out_feat_keys == None:

            return x1, x2, transform_output,classify_output,attention_matrix1,attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2