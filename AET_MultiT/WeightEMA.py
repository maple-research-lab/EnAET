# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
from Model.Classifier_Mixmatch import Classifier_Mixmatch
from Model.Mixmatch_Wide_Resnet import WX_WideResNet,WX_LargeWideResNet,WX_WideResNet_STL
from Model.TE_Module import TE_Module
from Model.Preact_Resnet import PreActResNet152,PreActResNet34,PreActResNet34STL
class WeightEMA(object):
    def __init__(self, model,learning_rate, ema_model, num_classes=10,alpha=0.999,run_type=0,dataset_name='cifar10'):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.run_type=run_type
        if self.run_type== 0:
            run_type1 = 0
            #use attention, here you can set to 1,
            # initially we tune parameter based on attention one but no difference
            if dataset_name!='STL10':
                self.tmp_model = WX_WideResNet(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.3,
                                               run_type=run_type1,
                                               num_stages=4)
            else:
                self.tmp_model = WX_WideResNet_STL(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.3, run_type=run_type1,
                                  num_stages=5)
        elif self.run_type==1:
            run_type1 = 0
            self.tmp_model = WX_LargeWideResNet(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.3,
                                           run_type=run_type1,
                                           num_stages=4)
        self.wd = 0.02 * learning_rate
        #init ema model with the self.model
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            # batchnorm stats do not belong to the model's parameters
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)#combine two weights step by step