# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from ops.utils import Logger
import os
from ops.os_operation import  mkdir
import torchvision.transforms as transforms
import datetime
import torchvision
import PIL
import time
import numpy as np
#from Data_Processing.AET5_Dataloader import AET5_Dataloader
from Data_Processing.AET_All_Dataloader import AET_All_Dataloader
from Model.Classifier_Mixmatch import Classifier_Mixmatch
from Model.Mixmatch_Wide_Resnet import WX_WideResNet,WX_LargeWideResNet,WX_WideResNet_STL
import torch.backends.cudnn as cudnn
#from Model.TE_AET_Model import TE_AET_Model
from Model.Wide_Resnet_AET import Wide_Resnet_AET_Model,Wide_Resnet_AET_LargeModel
from MixMatch.SemiLoss import SemiLoss
from AET_MultiT.WeightEMA import WeightEMA
from AET_MultiT.train_AETall_fast import train_AETAllfast
from AET_MultiT.Val_AETAll_fast import Val_AETAll_fast
#from MixMatch.WeightEMA import WeightEMA
#from MixMatch.train_Mixmatch import train_Mixmatch
#from MixMatch.Val_Mixmatch import Val_Mixmatch
#from AET_MultiT.train_AET5 import train_AET5
#from AET_MultiT.Val_AET5 import Val_AET5
#from ops.Plot_Mixmatch_Log import Plot_Mixmatch_Log
from MixMatch.Adjust_LR import update_AET_LR,update_CLF_LR,update_AET_LRnew,update_AET_LR_prev,update_AET_LR_1000,update_AET_LR_Large
#from AET_MultiT.WeightSWA import WeightSWA
from AET_MultiT.train_AETAll4 import train_AETAll4
#from AET_MultiT.train_AETAll import train_AETAll#slower version, we dit not use it anymore, but they work same as fast version
#from AET_MultiT.Val_AETAll import Val_AETAll
#from Model.TE_Module import TE_Module
#from Model.Preact_Resnet import PreActResNet152,PreActResNet34,PreActResNet34STL
#from Model.Preact_Resnet_AETModule import Preact_Resnet_AETModule
from MixMatch.Adjust_LR import update_SVHNextra
def create_model(type,num_classes,dataset_name,ema=False):
    if type == 0:
        #control use attention or not
        run_type=0#use attention, here you can set to 1,
            # initially we tune parameter based on attention one but no difference
        #change dropout rate
        if dataset_name=='STL10':
            #8 2080ti are needed
            model=WX_WideResNet_STL(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.0,run_type=run_type,num_stages=5)

        else:
            #1 2080ti enough
            model = WX_WideResNet(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.0, run_type=run_type,
                                  num_stages=4)


    elif type==1:
        #large wideresnet-28*2 using filter size=135
        run_type = 0#close attention module
        # change dropout rate
        #4 2080ti needed
        #this is 26M parameters
        model = WX_LargeWideResNet(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.0, run_type=run_type,
                              num_stages=4)


    if ema:
        for param in model.parameters():
            param.detach_()

    return model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def Generate_Mixmatch5AET_Improved_Model(data,params):
    np.random.seed(params['seed'])
    type = params['type']
    log_path, result_path = Get_log_path(params)

    dataset_name = params['dataset']
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name=='SVHN':
        num_classes=10
    elif dataset_name=='STL10':
        num_classes=10
    elif dataset_name=='SVHNextra':
        num_classes=10
    if dataset_name=='SVHNextra':#we don't have time to run this in the end
        train_dataloader, unlabel_dataloader, testloader = prepare_SVHN_extra_Dataloader(data, params, num_classes)
    elif dataset_name!='STL10':
        train_dataloader, unlabel_dataloader, testloader = prepare_Dataloader(data, params, num_classes)
    else:
        #specific dataloader that can directly load data into memory to reduce computation on cpu
        train_dataloader,unlabel_dataloader,testloader=prepare_memory_Dataloader(data,params,num_classes)
    model = create_model(type, num_classes,dataset_name)
    ema_model = create_model(type, num_classes,dataset_name, ema=True)
    #remove bad performance of swa-model
    #swa_model = create_model(type,num_classes,ema=True)#ema used to contral grad or not,here no grad

    assert type==0 or type==1
    if type==0:
        if dataset_name!='STL10':
            aet_model1 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=8, dropRate=0.0, widen_factor=2)
            aet_model2 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=6, dropRate=0.0, widen_factor=2)
            aet_model3 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=5, dropRate=0.0, widen_factor=2)
            aet_model4 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=4, dropRate=0.0, widen_factor=2)
            aet_model5 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=4, dropRate=0.0, widen_factor=2)
        else:#for STL10
            aet_model1 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=8, dropRate=0.0, widen_factor=2,run_type=2)
            aet_model2 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=6, dropRate=0.0, widen_factor=2,run_type=2)
            aet_model3 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=5, dropRate=0.0, widen_factor=2,run_type=2)
            aet_model4 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=4, dropRate=0.0, widen_factor=2,run_type=2)
            aet_model5 = Wide_Resnet_AET_Model(num_layer_in_block=4, num_classes=4, dropRate=0.0, widen_factor=2,run_type=2)
    elif type==1:
        aet_model1 = Wide_Resnet_AET_LargeModel(num_layer_in_block=4, num_classes=8, dropRate=0.0, widen_factor=2)
        aet_model2 = Wide_Resnet_AET_LargeModel(num_layer_in_block=4, num_classes=6, dropRate=0.0, widen_factor=2)
        aet_model3 = Wide_Resnet_AET_LargeModel(num_layer_in_block=4, num_classes=5, dropRate=0.0, widen_factor=2)
        aet_model4 = Wide_Resnet_AET_LargeModel(num_layer_in_block=4, num_classes=4, dropRate=0.0, widen_factor=2)
        aet_model5 = Wide_Resnet_AET_LargeModel(num_layer_in_block=4, num_classes=4, dropRate=0.0, widen_factor=2)

    cudnn.benchmark = True
    print('    Total params: %.10fM' % (count_parameters(model) / 1000000.0))
    print('AET1 part params: %.10fM' % (count_parameters(aet_model1) / 1000000.0))
    print('AET2 part params: %.10fM' % (count_parameters(aet_model2) / 1000000.0))
    print('AET3 part params: %.10fM' % (count_parameters(aet_model3) / 1000000.0))
    print('AET4 part params: %.10fM' % (count_parameters(aet_model4) / 1000000.0))
    print('AET5 part params: %.10fM' % (count_parameters(aet_model5) / 1000000.0))
    # if dataset_name == 'cifar10' or dataset_name == 'cifar100':
    #     #if type!=4:
    #     #    mixmatch_criterion = SemiLoss(params)
    #     #else:
    #         mixmatch_criterion = SemiLoss(params)
    # elif dataset_name=='SVHN':
    #     mixmatch_criterion=SemiLoss(params,1024)
    # elif dataset_name=='STL10':
    mixmatch_criterion = SemiLoss(params,params['Mixmatch_warm'])#first try simple warmup
    clf_criterion = nn.CrossEntropyLoss()
    aet_criterion = nn.MSELoss()
    if params['cuda']:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        ema_model = ema_model.cuda()
        ema_model = nn.DataParallel(ema_model, device_ids=None)
        clf_criterion = clf_criterion.cuda()
        aet_model1 = aet_model1.cuda()
        aet_model1 = nn.DataParallel(aet_model1, device_ids=None)
        aet_model2 = aet_model2.cuda()
        aet_model2 = nn.DataParallel(aet_model2, device_ids=None)
        aet_model3 = aet_model3.cuda()
        aet_model3 = nn.DataParallel(aet_model3, device_ids=None)
        aet_model4 = aet_model4.cuda()
        aet_model4 = nn.DataParallel(aet_model4, device_ids=None)
        aet_model5 = aet_model5.cuda()
        aet_model5 = nn.DataParallel(aet_model5, device_ids=None)
        aet_criterion = aet_criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    aet_optimizer1 = optim.SGD(
        aet_model1.parameters(),
        lr=params['lr1'],
        momentum=params['momentum'],  # Use default
        # dampening=0.9,#Nesterov momentum requires a momentum and zero dampening
        weight_decay=params['weight_decay'], nesterov=True)  # Use default
    aet_optimizer2 = optim.SGD(
        aet_model2.parameters(),
        lr=params['lr1'],
        momentum=params['momentum'],  # Use default
        # dampening=0.9,#Nesterov momentum requires a momentum and zero dampening
        weight_decay=params['weight_decay'], nesterov=True)  # Use default
    aet_optimizer3 = optim.SGD(
        aet_model3.parameters(),
        lr=params['lr1'],
        momentum=params['momentum'],  # Use default
        # dampening=0.9,#Nesterov momentum requires a momentum and zero dampening
        weight_decay=params['weight_decay'], nesterov=True)  # Use default
    aet_optimizer4 = optim.SGD(
        aet_model4.parameters(),
        lr=params['lr1'],
        momentum=params['momentum'],  # Use default
        # dampening=0.9,#Nesterov momentum requires a momentum and zero dampening
        weight_decay=params['weight_decay'], nesterov=True)  # Use default
    aet_optimizer5 = optim.SGD(
        aet_model5.parameters(),
        lr=params['lr1'],
        momentum=params['momentum'],  # Use default
        # dampening=0.9,#Nesterov momentum requires a momentum and zero dampening
        weight_decay=params['weight_decay'], nesterov=True)  # Use default
    ema_optimizer = WeightEMA(model, params['lr'], ema_model, num_classes=num_classes, alpha=params['ema_decay'],
                              run_type=type,dataset_name=dataset_name)
    #swa_optimizer = WeightSWA(model, swa_model, num_classes=num_classes,run_type=type)
    start_epoch = params['start_epoch']
    train_logger, train_batch_logger, Label_Logger, Val_logger, Test_logger,SWA_logger = init_Logger(log_path)

    if params['resume']==1:
        model_path = os.path.abspath(params['M'])
        pretrain = torch.load(model_path)
        #load_fromprev=pretrain['state_dict']
        #for name in load_fromprev:
        #    print(name)
        model.load_state_dict(pretrain['state_dict'])
        ema_model.load_state_dict(pretrain['ema_state_dict'])
        #if start_epoch>params['swa_start_epoch']:
         #   swa_model.load_state_dict(pretrain['swa_state_dict'])
        aet_model1.load_state_dict(pretrain['aet1_state_dict'])
        aet_model2.load_state_dict(pretrain['aet2_state_dict'])
        aet_model3.load_state_dict(pretrain['aet3_state_dict'])
        aet_model4.load_state_dict(pretrain['aet4_state_dict'])
        aet_model5.load_state_dict(pretrain['aet5_state_dict'])
        #for some adam case, loading the optimizer also
        optimizer.load_state_dict(pretrain['optimizer'])
        #still need to reload the optimizer again because of the momentum buffer in SGD. Otherwise, it will lose that.
        aet_optimizer1.load_state_dict(pretrain['aet1_optimizer'])
        aet_optimizer2.load_state_dict(pretrain['aet2_optimizer'])
        aet_optimizer3.load_state_dict(pretrain['aet3_optimizer'])
        aet_optimizer4.load_state_dict(pretrain['aet4_optimizer'])
        aet_optimizer5.load_state_dict(pretrain['aet5_optimizer'])

    best_acc = 0
    val_acc=0
    test_accs = []
    test_acc=0#record here for start with odd epoch, such as 91
    iteration = params['mix_iteration']
    print('iteration %d' % iteration)
    writer = None
    if params['tensorboard']:
        from tensorboardX import SummaryWriter
        log_tensor = os.path.join(log_path, 'Tensorboard')
        writer = SummaryWriter(log_tensor)
        writer.add_text('Text', str(params))
    aet_model=[aet_model1,aet_model2,aet_model3,aet_model4,aet_model5]
    aet_optimizer=[aet_optimizer1,aet_optimizer2,aet_optimizer3,aet_optimizer4,aet_optimizer5]
    for i in range(start_epoch, params['epochs']):

        # if type==0:
        #     if dataset_name!='STL10':
        update_AET_LR(i,aet_optimizer1)#will not influence too much, thus we updated it
        update_AET_LR(i, aet_optimizer2)
        update_AET_LR(i, aet_optimizer3)
        update_AET_LR(i, aet_optimizer4)
        update_AET_LR(i, aet_optimizer5)
        #     else:
        #         update_AET_LR_Large(i, aet_optimizer1)#stl10 actually based super big model
        #         update_AET_LR_Large(i, aet_optimizer2)
        #         update_AET_LR_Large(i, aet_optimizer3)
        #         update_AET_LR_Large(i, aet_optimizer4)
        #         update_AET_LR_Large(i, aet_optimizer5)
        #
        # elif type==1:
        #     update_AET_LR_Large(i,aet_optimizer1)
        #     update_AET_LR_Large(i, aet_optimizer2)
        #     update_AET_LR_Large(i, aet_optimizer3)
        #     update_AET_LR_Large(i, aet_optimizer4)
        #     update_AET_LR_Large(i, aet_optimizer5)
        if params['max_lambda4']==0:
            #to save gpu memory
            train_AETAll4(train_dataloader, unlabel_dataloader, model, optimizer, ema_optimizer, mixmatch_criterion, i,
                           params['cuda'], aet_model,True,aet_criterion,aet_optimizer,iteration,
                           params['T'],params['alpha'],params['lambda'],train_logger,train_batch_logger,
                           type,writer,num_classes,params,ema_model)
        else:
            train_AETAllfast(train_dataloader, unlabel_dataloader, model, optimizer, ema_optimizer, mixmatch_criterion, i,
                           params['cuda'], aet_model,True,aet_criterion,aet_optimizer,iteration,
                           params['T'],params['alpha'],params['lambda'],train_logger,train_batch_logger,
                           type,writer,num_classes,params,ema_model)

        if i % 10 == 0 or (i >= 500 and i % 2 == 0) or i >= 1000:
            if params['use_ema']:
                label_loss, label_acc = Val_AETAll_fast(testloader,model, clf_criterion, i,
                                                            params['cuda'], aet_model, True, aet_criterion,
                                                            params, Label_Logger,writer,'Model-test')
            if dataset_name!='STL10' and dataset_name!="SVHNextra":#ignore this because unlabeled dataset do not have labels, all recorded as -1
                if params['use_ema']:
                    val_loss,val_acc=Val_AETAll_fast(unlabel_dataloader, ema_model, clf_criterion, i,
                         params['cuda'], aet_model, True, aet_criterion,
                         params, Val_logger,writer,'EMA-val')
                else:
                    val_loss, val_acc = Val_AETAll_fast(unlabel_dataloader, model, clf_criterion, i,
                                                        params['cuda'], aet_model, True, aet_criterion,
                                                        params, Val_logger, writer, 'EMA-val')
        if i%2==0 or i>=500:
            if params['use_ema']:
                test_loss,test_acc = Val_AETAll_fast(testloader, ema_model, clf_criterion, i,
                                               params['cuda'], aet_model, True, aet_criterion,
                                               params, Test_logger,writer,'EMA-test')
            else:
                test_loss, test_acc = Val_AETAll_fast(testloader, model, clf_criterion, i,
                                                      params['cuda'], aet_model, True, aet_criterion,
                                                      params, Test_logger, writer, 'EMA-test')
        #validation and testing

        if i>=1000:
            save_file_path = os.path.join(result_path,
                                          'save_{}.pth'.format(i))

            states = {
                    'epoch': i + 1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'aet1_state_dict':aet_model1.state_dict(),
                    'aet1_optimizer':aet_optimizer1.state_dict(),
                    'aet2_state_dict': aet_model2.state_dict(),
                    'aet2_optimizer': aet_optimizer2.state_dict(),
                    'aet3_state_dict': aet_model3.state_dict(),
                    'aet3_optimizer': aet_optimizer3.state_dict(),
                    'aet4_state_dict': aet_model4.state_dict(),
                    'aet4_optimizer': aet_optimizer4.state_dict(),
                    'aet5_state_dict': aet_model5.state_dict(),
                    'aet5_optimizer': aet_optimizer5.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            torch.save(states, save_file_path)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            #    'swa_state_dict': swa_model.state_dict(),
            'acc': val_acc,
                'aet1_state_dict': aet_model1.state_dict(),
                'aet1_optimizer': aet_optimizer1.state_dict(),
                'aet2_state_dict': aet_model2.state_dict(),
                'aet2_optimizer': aet_optimizer2.state_dict(),
                'aet3_state_dict': aet_model3.state_dict(),
                'aet3_optimizer': aet_optimizer3.state_dict(),
                'aet4_state_dict': aet_model4.state_dict(),
                'aet4_optimizer': aet_optimizer4.state_dict(),
                'aet5_state_dict': aet_model5.state_dict(),
                'aet5_optimizer': aet_optimizer5.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            }, is_best,checkpoint=result_path)
        if i % 2 == 0 or i >= 500:
            test_accs.append(test_acc)
    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))
    if writer!=None:
        writer.close()
import os
import shutil
def save_checkpoint(state, is_best, checkpoint,filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def Get_log_path(params):
    learning_rate = params['lr']
    learning_rate1 = params['lr1']
    reg = params['reg']
    type = params['type']
    lambda1 = params['lambda']
    lambda2 = params['lambda1']
    lambda3 = params['lambda2']
    lambda4 = params['lambda3']
    lambda5 = params['lambda4']
    if params['S']=='':
        log_path = os.path.join(os.getcwd(), params['log_path'])
    else:
        log_path= os.path.join(params['S'], params['log_path'])
    mkdir(log_path)
    dataset_name=params['dataset']
    log_path=os.path.join(log_path,dataset_name)
    mkdir(log_path)
    if type == 0:
        log_path = os.path.join(log_path, 'MixMatch-5AETNEW')
    elif type==1:
        log_path = os.path.join(log_path, 'MixMatch-5AETNEW-Wideresnet')
    elif type==2:
        log_path = os.path.join(log_path, 'MixMatch-5AETNEW-Wideresnet500')
    elif type==3:
        log_path = os.path.join(log_path, 'MixMatch-5AETNEW-Wideresnet1000')
    elif type==4:
        log_path = os.path.join(log_path, 'MixMatch-5AETNEW-LargeWideresnet')
    elif type == 5:
        log_path = os.path.join(log_path, 'MixMatch-5AETNEW-Resnet152')
    mkdir(log_path)
    portion = params['portion']
    log_path = os.path.join(log_path, 'Label_portion' + str(portion))
    mkdir(log_path)
    log_path = os.path.join(log_path, 'lr_' + str(learning_rate) + 'lr1_' + str(learning_rate1) + '_reg_' + str(reg))
    mkdir(log_path)
    log_path = os.path.join(log_path, 'lambda_' + str(lambda1)+'-'+ str(lambda2)+'-'+ str(lambda3)+'-'+ str(lambda4)+'-'+ str(lambda5))
    mkdir(log_path)
    log_path = os.path.join(log_path, 'beta_' + str(params['beta']))
    mkdir(log_path)
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    log_path = os.path.join(log_path, formatted_today + now)
    mkdir(log_path)
    result_path = os.path.join(log_path,'model')
    mkdir(result_path)
    return log_path,result_path

def prepare_memory_Dataloader(data,params,num_classes):
    from Data_Processing.AET_Memory_Dataset import AET_Memory_Dataloader
    if params['dataset']=='STL10':
        #channel first
        #only based on the training dataset not unlabel dataset
        #TRAIN_MEAN = (0.44671062065972217, 0.43980983983523964, 0.40664644709967324)  # calculate by myself
        #TRAIN_STD = (0.26034097826623354, 0.2565772731134436, 0.2712673814522548)
        TRAIN_MEAN=(0.44087801806139126,0.42790631331699347,0.3867879370752931)
        TRAIN_STD=(0.26826768614218743,0.26104504009696305,0.2686683686297828)
    from ops.Transform_ops import RandomFlip,RandomPadandCrop
    transform_train = transforms.Compose([
        RandomPadandCrop(96,pad_size=12),
        RandomFlip(),
    ])
    """
    in cutout paper:
    We perform a grid search over the cutout size parameter using 10% of the training images as a validation set
and select a square size of 24 × 24 pixels for the no dataaugmentation case and 32 × 32 pixels for training STL-10
with data augmentation
    """
    unlabel_dataset = AET_Memory_Dataloader(dataset_dir=data.train_path, dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD,shift=params['shift'],degrees=params['rot'],shear=params['shear'],
                                                    train_label=True,translate=(params['translate'], params['translate']),
                                                    scale=(params['shrink'], params['enlarge']),
                                                    fillcolor=(128, 128, 128),
                                                    resample=PIL.Image.BILINEAR,
                                                    matrix_transform=transforms.Compose([
                                                        transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                             (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                                    ]),
                                                    transform_pre= transform_train, rand_state=params['seed'],
                                                    valid_size=0, num_classes=num_classes,extra_path=data.extra_path,patch_length=24,
                                                    )
    unlabel_dataloader = torch.utils.data.DataLoader(unlabel_dataset, batch_size=params['batch_size'],
                                                   shuffle=True, num_workers=int(params['num_workers']),
                                                     drop_last=True,pin_memory=True)
    train_labeled_dataset = AET_Memory_Dataloader(dataset_dir=data.train_path,dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD, shift=params['shift'],degrees=params['rot'],shear=params['shear'],
                                                   train_label=False,translate=(params['translate'], params['translate']),
                                                   scale=(params['shrink'], params['enlarge']),
                                                   fillcolor=(128, 128, 128), resample=PIL.Image.BILINEAR,
                                                   matrix_transform=transforms.Compose([
                                                       transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                            (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                                   ]),
                                                   transform_pre= transform_train, rand_state=params['seed'],
                                                    valid_size=params['portion'], uniform_label=True,
                                                            num_classes=num_classes,patch_length=24,
                                                   )
    train_dataloader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=params['batch_size'],
                                                   shuffle=True, num_workers=int(params['num_workers']),drop_last=True,pin_memory=True)
    test_dataset = AET_Memory_Dataloader(dataset_dir=data.test_path, dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD,shift=params['shift'], train_label=True,
                                                   degrees=params['rot'], shear=params['shear'],translate=(params['translate'], params['translate']),
                                          scale=(params['shrink'], params['enlarge']), fillcolor=(128, 128, 128),
                                          resample=PIL.Image.BILINEAR,
                                          matrix_transform=transforms.Compose([
                                              transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                   (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                          ]),
                                          rand_state=params['seed'], valid_size=0,num_classes=num_classes,patch_length=24,
                                          )
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                             num_workers=int(params['num_workers']),pin_memory=True)
    return train_dataloader,unlabel_dataloader,testloader

def prepare_Dataloader(data,params,num_classes):
    if params['dataset']=='cifar10':
        TRAIN_MEAN = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        TRAIN_STD= (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255
    elif params['dataset']=='cifar100':
        TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif params['dataset']=='SVHN':#method verified on the cifar10
        TRAIN_MEAN = (0.4376821046090723,0.4437697045639686,0.4728044222297267)#calculate by myself
        TRAIN_STD = (0.19803012447157134,0.20101562471828877,0.19703614172172396)
    #dataset=torchvision.dataset.cifar10(params['F'], train=True, download=True)
    from ops.Transform_ops import RandomFlip,RandomPadandCrop
    transform_train = transforms.Compose([
        RandomPadandCrop(32),
        RandomFlip(),
        # dataset.ToTensor(),
    ])

    unlabel_dataset = AET_All_Dataloader(dataset_dir=data.train_path, dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD,shift=params['shift'],degrees=params['rot'],shear=params['shear'],
                                                    train_label=True,translate=(params['translate'], params['translate']),
                                                    scale=(params['shrink'], params['enlarge']),
                                                    fillcolor=(128, 128, 128),
                                                    resample=PIL.Image.BILINEAR,
                                                    matrix_transform=transforms.Compose([
                                                        transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                             (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                                    ]),
                                                    transform_pre= transform_train, rand_state=params['seed'],
                                                    valid_size=0, num_classes=num_classes,
                                                    )
    unlabel_dataloader = torch.utils.data.DataLoader(unlabel_dataset, batch_size=params['batch_size'],
                                                   shuffle=True, num_workers=int(params['num_workers']),
                                                     drop_last=True,pin_memory=True)
    train_labeled_dataset = AET_All_Dataloader(dataset_dir=data.train_path,dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD, shift=params['shift'],degrees=params['rot'],shear=params['shear'],
                                                   train_label=False,translate=(params['translate'], params['translate']),
                                                   scale=(params['shrink'], params['enlarge']),
                                                   fillcolor=(128, 128, 128), resample=PIL.Image.BILINEAR,
                                                   matrix_transform=transforms.Compose([
                                                       transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                            (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                                   ]),
                                                   transform_pre= transform_train, rand_state=params['seed'],
                                                    valid_size=params['portion'], uniform_label=True,
                                                            num_classes=num_classes,
                                                   )
    train_dataloader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=params['batch_size'],
                                                   shuffle=True, num_workers=int(params['num_workers']),drop_last=True,pin_memory=True)
    test_dataset = AET_All_Dataloader(dataset_dir=data.test_path, dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD,shift=params['shift'], train_label=True,
                                                   degrees=params['rot'], shear=params['shear'],translate=(params['translate'], params['translate']),
                                          scale=(params['shrink'], params['enlarge']), fillcolor=(128, 128, 128),
                                          resample=PIL.Image.BILINEAR,
                                          matrix_transform=transforms.Compose([
                                              transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                   (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                          ]),
                                          rand_state=params['seed'], valid_size=0,num_classes=num_classes,
                                          )
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                             num_workers=int(params['num_workers']),pin_memory=True)
    return train_dataloader,unlabel_dataloader,testloader
def prepare_SVHN_extra_Dataloader(data, params, num_classes):
    TRAIN_MEAN = (0.4376821046090723,0.4437697045639686,0.4728044222297267)#calculate by myself
    TRAIN_STD = (0.19803012447157134,0.20101562471828877,0.19703614172172396)
    #dataset=torchvision.dataset.cifar10(params['F'], train=True, download=True)
    from ops.Transform_ops import RandomFlip,RandomPadandCrop
    transform_train = transforms.Compose([
        RandomPadandCrop(32),
        RandomFlip(),
        # dataset.ToTensor(),
    ])

    unlabel_dataset = AET_All_Dataloader(dataset_dir=data.train_path, dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD,shift=params['shift'],degrees=params['rot'],shear=params['shear'],
                                                    train_label=True,translate=(params['translate'], params['translate']),
                                                    scale=(params['shrink'], params['enlarge']),
                                                    fillcolor=(128, 128, 128),
                                                    resample=PIL.Image.BILINEAR,
                                                    matrix_transform=transforms.Compose([
                                                        transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                             (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                                    ]),
                                                    transform_pre= transform_train, rand_state=params['seed'],
                                                    valid_size=0, num_classes=num_classes,extra_path=data.extra_path
                                                    )
    unlabel_dataloader = torch.utils.data.DataLoader(unlabel_dataset, batch_size=params['batch_size'],
                                                   shuffle=True, num_workers=int(params['num_workers']),
                                                     drop_last=True,pin_memory=True)
    train_labeled_dataset = AET_All_Dataloader(dataset_dir=data.train_path,dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD, shift=params['shift'],degrees=params['rot'],shear=params['shear'],
                                                   train_label=False,translate=(params['translate'], params['translate']),
                                                   scale=(params['shrink'], params['enlarge']),
                                                   fillcolor=(128, 128, 128), resample=PIL.Image.BILINEAR,
                                                   matrix_transform=transforms.Compose([
                                                       transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                            (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                                   ]),
                                                   transform_pre= transform_train, rand_state=params['seed'],
                                                    valid_size=params['portion'], uniform_label=True,
                                                            num_classes=num_classes,
                                                   )
    train_dataloader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=params['batch_size'],
                                                   shuffle=True, num_workers=int(params['num_workers']),drop_last=True,pin_memory=True)
    test_dataset = AET_All_Dataloader(dataset_dir=data.test_path, dataset_mean=TRAIN_MEAN,dataset_std=TRAIN_STD,shift=params['shift'], train_label=True,
                                                   degrees=params['rot'], shear=params['shear'],translate=(params['translate'], params['translate']),
                                          scale=(params['shrink'], params['enlarge']), fillcolor=(128, 128, 128),
                                          resample=PIL.Image.BILINEAR,
                                          matrix_transform=transforms.Compose([
                                              transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.),
                                                                   (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                                          ]),
                                          rand_state=params['seed'], valid_size=0,num_classes=num_classes,
                                          )
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                             num_workers=int(params['num_workers']),pin_memory=True)
    print("Unlabelled dataset size %d"%len(unlabel_dataset))
    return train_dataloader,unlabel_dataloader,testloader

def init_Logger(log_path):

    train_logger = Logger(
        os.path.join(log_path, 'train.log'),
        ['epoch', 'loss', 'AET1', 'AET2','AET3','AET4','AET5','Closs', 'Eloss','KL','wd','waet','top1', 'top5', 'lr1','lr2'])
    train_batch_logger = Logger(
        os.path.join(log_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss',  'AET1','AET2','AET3','AET4', 'AET5','Closs', 'Eloss', 'KL','wd','waet','top1', 'top5', 'lr1','lr2'])
    Label_Logger=Logger(
        os.path.join(log_path, 'trainlabel.log'), ['epoch',  'AET1','AET2','AET3','AET4', 'AET5', 'Closs', 'top1','top1avg', 'top5'])
    Val_logger = Logger(
        os.path.join(log_path, 'val.log'), ['epoch',  'AET1','AET2','AET3','AET4','AET5',  'Closs',  'top1', 'top1avg','top5'])
    Test_logger = Logger(
        os.path.join(log_path, 'test.log'), ['epoch', 'AET1','AET2','AET3','AET4', 'AET5', 'Closs','top1','top1avg', 'top5'])
    SWA_logger = Logger(
        os.path.join(log_path, 'SWAtest.log'), ['epoch', 'AET1', 'AET2', 'AET3', 'AET4', 'AET5','Closs',  'top1','top1avg', 'top5'])
    return train_logger,train_batch_logger,Label_Logger,Val_logger,Test_logger,SWA_logger