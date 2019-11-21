# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import math
def update_AET_LR_1000(epoch,optimizer):
    """
    drop this because the performance is limited
    :param epoch:
    :param optimizer:
    :return:
    """
    if epoch<=1000:
        eta_min=0.001
        T_max=1000
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * epoch / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 1) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=1500:
        eta_min = 0.0001
        T_max = 500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 1000) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1001) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=2000:
        eta_min = 0.00001
        T_max = 500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 1500) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1501) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
def update_AET_LR(epoch,optimizer):
    if epoch<=500:
        eta_min=0.01
        T_max=500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * epoch / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 1) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=1000:
        eta_min=0.001
        T_max=500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch-500) / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 501) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=1300:
        eta_min = 0.0001
        T_max = 300
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 1000) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1001) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=1500:
        eta_min = 0.00001
        T_max = 200
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 1300) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1301) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
def update_AET_LR_Large(epoch,optimizer):
    if epoch<=500:
        eta_min=0.01
        T_max=500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * epoch / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 1) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=800:
        eta_min=0.001
        T_max=300
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch-500) / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 501) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=900:
        eta_min = 0.0001
        T_max = 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 800) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 801) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=1000:
        eta_min = 0.00001
        T_max = 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 900) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 901) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min

def update_SVHNextra(epoch,optimizer):
    if epoch<=250:
        eta_min=0.01
        T_max=250
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * epoch / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 1) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=370:
        eta_min=0.001
        T_max=120
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch-250) / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 251) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=450:
        eta_min = 0.0001
        T_max = 80
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 370) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 371) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=500:
        eta_min = 0.00001
        T_max = 50
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 450) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 451) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min


def update_AET_LR_prev(epoch,optimizer):
    """
    drop this because the performance is limited
    :param epoch:
    :param optimizer:
    :return:
    """
    if epoch<=500:
        eta_min=0.001
        T_max=500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * epoch / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 1) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=1000:
        eta_min = 0.0001
        T_max = 500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 500) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 501) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=1500:
        eta_min = 0.00001
        T_max = 500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 1000) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1001) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
def update_AET_LRnew(epoch,optimizer):
    if epoch<=1000:
        eta_min=0.001
        T_max=1000
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * epoch / T_max)) /\
            (1 + math.cos(math.pi * (epoch - 1) / T_max)) *\
                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=1500:
        eta_min = 0.0001
        T_max = 500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 1000) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1001) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
    elif epoch<=2000:
        eta_min = 0.00001
        T_max = 500
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch - 1500) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1501) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min

def update_CLF_LR(epoch, optimizer):
    if epoch <= 800:
        eta_min = 0.001
        T_max = 1000
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * epoch / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min
    elif epoch <= 1500:
        eta_min = 0.0001
        T_max = 1000
        for param_group in optimizer.param_groups:
            param_group['lr'] = (1 + math.cos(math.pi * (epoch-1500) / T_max)) / \
                                (1 + math.cos(math.pi * (epoch - 1501) / T_max)) * \
                                (param_group['lr'] - eta_min) + eta_min

