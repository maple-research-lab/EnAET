from Model.NIN import NetworkInNetwork,NetworkInNetwork1,NetworkInNetwork2
import torch
import torch.nn as nn
from Model.Cross_Attention import Cross_Attn
from Model.Basic_Block import GlobalAveragePooling
from Model.Cross_NIN import Cross_NIN
class Regressor(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True, indim=384, num_classes=6):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        #nChannels = 192
        super(Regressor, self).__init__()
        self.nin = NetworkInNetwork(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3)
        self.fc = nn.Linear(indim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1, x2, out_feat_keys=None):
        x1,attention_matrix1 = self.nin(x1, out_feat_keys)
        x2,attention_matrix2 = self.nin(x2, out_feat_keys)
        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)
        if out_feat_keys == None:
            x = torch.cat((x1, x2), dim=1)
            return x1, x2, torch.tanh(self.fc(x)),attention_matrix1,attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2

class Cross_Regressor(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True, indim=384, num_classes=6):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        nChannels = 192
        super(Cross_Regressor, self).__init__()
        self.num_stages=_num_stages
        self.nin = NetworkInNetwork(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3)
        self.fc = nn.Linear(indim, num_classes)
        self.cross_Attn=Cross_Attn(nChannels,'relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.pool_operation=GlobalAveragePooling()
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1, x2, out_feat_keys=None):
        if out_feat_keys == 'Attention00':
            #cross attention+self attention
            out_feat_key1='Attention'
            specific_key='conv'+str(self.num_stages)
            output_key='classifier'
            feat_list1,attention_matrix1=self.nin(x1, [specific_key,out_feat_key1,output_key])
            feat_list2, attention_matrix2 = self.nin(x2, [specific_key, out_feat_key1,output_key])
            #then do cross-attention for the feat_list[0],output from the conv blocks
            feat1=feat_list1[0]
            feat2=feat_list2[0]
            cross_feat1=self.cross_Attn(feat1,attention_matrix2)
            cross_feat2=self.cross_Attn(feat2,attention_matrix1)
            cross_feat1=self.pool_operation(cross_feat1)
            cross_feat2=self.pool_operation(cross_feat2)
            #concat and predict
            feat1=feat_list1[-1]
            feat2=feat_list2[-1]
        elif out_feat_keys == 'Attention01':#only use cross attention
            out_feat_key1 = 'Attention'
            specific_key = 'conv' + str(self.num_stages)
            output_key = 'classifier'
            feat_list1, attention_matrix1 = self.nin(x1, [specific_key, out_feat_key1, output_key])
            feat_list2, attention_matrix2 = self.nin(x2, [specific_key, out_feat_key1, output_key])
            feat1 = feat_list1[0]
            feat2 = feat_list2[0]
            cross_feat1 = self.cross_Attn(feat1, attention_matrix2)
            cross_feat2 = self.cross_Attn(feat2, attention_matrix1)
            cross_feat1 = self.pool_operation(cross_feat1)
            cross_feat2 = self.pool_operation(cross_feat2)
            x1=cross_feat1
            x2=cross_feat2
        else:
            x1,attention_matrix1 = self.nin(x1, out_feat_keys)
            x2,attention_matrix2 = self.nin(x2, out_feat_keys)


        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)
        if out_feat_keys == None:
            x = torch.cat((x1, x2), dim=1)
            return x1, x2, torch.tanh(self.fc(x)),attention_matrix1,attention_matrix2
        elif out_feat_keys=="Attention00":
            x = torch.cat((feat1,feat2,cross_feat1,cross_feat2), dim=1)
            #print(x.size())
            return x1, x2, cross_feat1,cross_feat2,torch.tanh(self.fc(x)), attention_matrix1, attention_matrix2
        elif out_feat_keys=="Attention01":
            x = torch.cat((x1, x2), dim=1)
            return x1, x2, torch.tanh(self.fc(x)), attention_matrix1, attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2
class Regressor1(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True, indim=384, num_classes=6):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        #nChannels = 192
        super(Regressor1, self).__init__()
        self.nin = NetworkInNetwork1(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3)
        self.fc = nn.Linear(indim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1, x2, out_feat_keys=None):
        x1,attention_matrix1 = self.nin(x1, out_feat_keys)
        x2,attention_matrix2 = self.nin(x2, out_feat_keys)
        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)
        if out_feat_keys == None:
            x = torch.cat((x1, x2), dim=1)
            return x1, x2, torch.tanh(self.fc(x)),attention_matrix1,attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2


class Regressor2(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True, indim=384, num_classes=6):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        #nChannels = 192
        super(Regressor2, self).__init__()
        self.nin = NetworkInNetwork2(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3)
        self.fc = nn.Linear(indim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1, x2, out_feat_keys=None):
        x1,attention_matrix1 = self.nin(x1, out_feat_keys)
        x2,attention_matrix2 = self.nin(x2, out_feat_keys)
        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)
        if out_feat_keys == None:
            x = torch.cat((x1, x2), dim=1)
            return x1, x2, torch.tanh(self.fc(x)),attention_matrix1,attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2

class Cross_RegressorB(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True, indim=384, num_classes=6):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        nChannels = 192
        super(Cross_RegressorB, self).__init__()
        self.num_stages=_num_stages
        self.nin = Cross_NIN(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3)
        self.fc = nn.Linear(indim, num_classes)
        #self.cross_Attn=Cross_Attn(nChannels,'relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.pool_operation=GlobalAveragePooling()
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1, x2, out_feat_keys=None):
        first_stage_list=['conv1']+['conv2']+['Attention']
        check_flag=True
        for item in out_feat_keys:
            if item not in first_stage_list:
                check_flag=False
        if check_flag:
            #only use first stage: cross-attention trained result directly applied to itself
            x1, attention_matrix1 = self.nin(x1, out_feat_keys=out_feat_keys)
            x2, attention_matrix2 = self.nin(x2,out_feat_keys= out_feat_keys)
        elif "Attention00" in out_feat_keys:
        #applying using, to use cross attention to apply to the image
            out_feat_key1 = 'Attention'
            specific_key = 'conv2'
            feat_list1, attention_matrix1 = self.nin(x1, out_feat_keys=[specific_key, out_feat_key1])
            feat_list2, attention_matrix2 = self.nin(x2, out_feat_keys=[specific_key, out_feat_key1])
            feat1 = feat_list1[0]  # output from conv without attention applying
            feat2 = feat_list2[0]
            # apply cross attention here
            output_key = ['Cross']
            x1, _ = self.nin(feat1, input_attention=attention_matrix2, out_feat_keys=output_key)
            x2, _ = self.nin(feat2, input_attention=attention_matrix1, out_feat_keys=output_key)
        else:
            out_feat_key1 = 'Attention'
            specific_key = 'conv2'
            feat_list1, attention_matrix1 = self.nin(x1, out_feat_keys=[specific_key, out_feat_key1])
            feat_list2, attention_matrix2 = self.nin(x2, out_feat_keys=[specific_key, out_feat_key1])
            feat1 = feat_list1[0]#output from conv without attention applying
            feat2 = feat_list2[0]
            #apply cross attention here
            output_key=['classifier']
            x1,_=self.nin(feat1, input_attention=attention_matrix2,out_feat_keys=output_key)
            x2,_=self.nin(feat2, input_attention=attention_matrix1,out_feat_keys=output_key)



        #x1,attention_matrix1=self.attention(x1)
        #x2,attention_matrix2=self.attention(x2)
        if out_feat_keys == None:
            x = torch.cat((x1, x2), dim=1)
            return x1, x2, torch.tanh(self.fc(x)),attention_matrix1,attention_matrix2

        elif out_feat_keys=="Attention00":
            #x = torch.cat((x1, x2), dim=1)
            return x1, x2,attention_matrix1, attention_matrix2
        else:
            return x1, x2,attention_matrix1,attention_matrix2