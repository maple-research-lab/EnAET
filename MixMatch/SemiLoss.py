# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class SemiLoss(object):
    def __init__(self,params,rampup_length=16):
        self.params=params
        self.rampup_length=rampup_length
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, iteration):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))#cross entropy
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.params['beta'] * self.linear_rampup(iteration)#use beta for the ent min loss

    def linear_rampup(self,current):
        rampup_length=self.rampup_length
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / float(rampup_length), 0.0, 1.0)
            return float(current)