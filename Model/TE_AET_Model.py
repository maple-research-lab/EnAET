import torch
import torch.nn as nn
from Model.TEBlock import TEBlock,TEBlock1,GlobalAveragePooling
class TE_AET_Model(nn.Module):
    def __init__(self, input_Channel, num_classes=8):
        super(TE_AET_Model, self).__init__()
        assert input_Channel==256
        nChannels = 128
        nChannels1 = 256
        nChannels2 = 512
        self.block=nn.Sequential()
        self.block.add_module('Block3_ConvB1', TEBlock1(nChannels1, nChannels2, 3))
        self.block.add_module('Block3_ConvB2', TEBlock1(nChannels2, nChannels1, 1))
        self.block.add_module('Block3_ConvB3', TEBlock1(nChannels1, nChannels, 1))
        self.block.add_module('Global_average',GlobalAveragePooling())
        self.fc = nn.Linear(nChannels*2, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x1, x2):
        x1=self.block(x1)
        x2=self.block(x2)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)
class TE_AET_Model_Share(nn.Module):
    def __init__(self, input_Channel):
        super(TE_AET_Model_Share, self).__init__()
        assert input_Channel==256
        nChannels = 128
        nChannels1 = 256
        nChannels2 = 512
        self.block=nn.Sequential()
        self.block.add_module('Block3_ConvB1', TEBlock1(nChannels1, nChannels2, 3))
        self.block.add_module('Block3_ConvB2', TEBlock1(nChannels2, nChannels1, 1))
        self.block.add_module('Block3_ConvB3', TEBlock1(nChannels1, nChannels, 1))
        self.block.add_module('Global_average',GlobalAveragePooling())

    def forward(self, x):
        x=self.block(x)
        return x
class Linear_Module(nn.Module):
    def __init__(self, num_classes=8,input_channels=128):
        super(Linear_Module, self).__init__()
        self.fc = nn.Linear(input_channels* 2, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)