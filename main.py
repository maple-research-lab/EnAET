# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import os
import numpy as np
from ops.argparser import argparser
if __name__ == "__main__":
    params = argparser()
    print(params)

    if params['mode'] == 0:
        # implement my mixmatch+4AET
        data_path = params['F']

        dataset_name = params['dataset']
        if dataset_name == 'cifar10':
            from Data_Processing.Download_Cifar import CIFAR10

            data = CIFAR10(data_path)
        elif dataset_name=='cifar100':
            from Data_Processing.Download_Cifar import CIFAR100

            data = CIFAR100(data_path)
        elif dataset_name=='SVHN':
            from Data_Processing.Download_SVHN import SVHN
            data=SVHN(data_path)
        elif dataset_name=='STL10':
            from Data_Processing.Download_STL import STL10
            data = STL10(data_path)
        elif dataset_name=='SVHNextra':
            from Data_Processing.Download_SVHN import SVHN

            data = SVHN(data_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = params['choose']
        import torch

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        import random

        random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        # use the 2nd layer conv+attention and then predict: type=1 and attention=0
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
        if use_cuda:
            torch.cuda.manual_seed_all(params['seed'])
            # Generate mixmatch model and applying AET on it
            from AET_MultiT.AET5_Improved_Mixmatch import Generate_Mixmatch5AET_Improved_Model
            Generate_Mixmatch5AET_Improved_Model(data,params)







