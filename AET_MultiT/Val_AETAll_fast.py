# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import torch
from torch.autograd import Variable
import time
import os
import torch.nn.functional as F
import sys
from ops.utils import AverageMeter
from ops.utils import Calculate_top_accuracy
from Data_Processing.data_prefetcher import data_prefetcher
import numpy as np
def Val_AETAll_fast(valloader, model, criterion, epoch,
                 use_cuda, aet_model,use_aet,aet_criterion,
                 params,Val_logger,writer,Type_Label):
    model.eval()
    if use_aet:
        aet_model1 = aet_model[0]
        aet_model2 = aet_model[1]
        aet_model3 = aet_model[2]
        aet_model4 = aet_model[3]
        aet_model5 = aet_model[4]
        aet_model1.eval()
        aet_model2.eval()
        aet_model3.eval()
        aet_model4.eval()
        aet_model5.eval()
    batch_time = AverageMeter()
    AET_loss1 = AverageMeter()
    AET_loss2 = AverageMeter()
    AET_loss3 = AverageMeter()
    AET_loss4 = AverageMeter()
    AET_loss5 = AverageMeter()
    Entropy_Loss=AverageMeter()
    Closs = AverageMeter()  # loss_x in mixmatch
    accuracies = AverageMeter()
    avg_accuracies = AverageMeter()
    top5 = AverageMeter()
    end= time.time()
    data_time = AverageMeter()
    iteration=len(valloader)
    prefetcher = data_prefetcher(valloader)
    with torch.no_grad():
        for batch_idx in range(iteration):
            read_data=prefetcher.next()
            (img1, img1_u2), img2, img3, img4, img5,img6, img7,aff_para, transform_matrix,oper_params, target=read_data
            data_time.update(time.time() - end)
            if use_cuda:
                img1, target = img1.cuda(), target.cuda(non_blocking=True)
                img2 = img2.cuda()
                img3 = img3.cuda()
                img4 = img4.cuda()
                img5 = img5.cuda()
                img6 = img6.cuda()
                img7 = img7.cuda()
                if use_aet:
                    aff_para=aff_para.cuda()
                    transform_matrix = transform_matrix.cuda()
                    oper_params=oper_params.cuda()
            outputs = model(img1)
            loss = criterion(outputs, target)
            prec1, prec5 = Calculate_top_accuracy(outputs.data, target.data, topk=(1, 5))
            Closs.update(loss.item(), img1.size(0))
            accuracies.update(prec1.item(), img1.size(0))
            top5.update(prec5.item(), img1.size(0))
            outputs_u2 = model(img1_u2)
            outputs_u3 = model(img2)
            outputs_u4 = model(img3)
            outputs_u5 = model(img4)
            outputs_u6 = model(img5)
            outputs_u7 = model(img6)
            outputs_u8 = model(img7)
            p = (torch.softmax(outputs, dim=1) + torch.softmax(outputs_u2, dim=1) ) / 2  # mean the softmax maatrix
            pt = p ** (1 / params['T'])
            logits_u = pt / pt.sum(dim=1, keepdim=True)
            p_1 = F.softmax(logits_u, dim=1)
            loss_ent=0

            p = (torch.softmax(outputs, dim=1) + torch.softmax(outputs_u2, dim=1)+torch.softmax(outputs_u3, dim=1)
                 +torch.softmax(outputs_u4, dim=1)+torch.softmax(outputs_u5, dim=1)+torch.softmax(outputs_u6, dim=1)
            +torch.softmax(outputs_u7, dim=1)+torch.softmax(outputs_u8, dim=1)) / 8  # mean the softmax maatrix
            pt = p ** (1 / params['T'])
            logits_u = pt / pt.sum(dim=1, keepdim=True)
            prec1, prec5 = Calculate_top_accuracy(logits_u.data, target.data, topk=(1, 5))
            avg_accuracies.update(prec1.item(),p_1.size(0))
            #also report the sharpened prediction(Averaged prediction)
            if use_aet:
                use_key = ['Attention']
                #projective transformation
                feature_map1,atten1 = model(img1, use_key)
                feature_map2,atten2 = model(img2, use_key)
                pred_transform = aet_model1(feature_map1, feature_map2)
                transform_matrix = transform_matrix.view(-1, 8)
                aet_loss = aet_criterion(pred_transform, transform_matrix)
                AET_loss1.update(aet_loss.item(),img1.size(0))
                #affine transformation
                feature_map2, atten2 = model(img3, use_key)
                pred_transform = aet_model2(feature_map1, feature_map2)
                aet_loss = aet_criterion(pred_transform, aff_para)
                AET_loss2.update(aet_loss.item(), img1.size(0))
                # similarity transformation
                feature_map2, atten2 = model(img4, use_key)
                pred_transform = aet_model3(feature_map1, feature_map2)
                aet_loss = aet_criterion(pred_transform, aff_para[:,:5])
                AET_loss3.update(aet_loss.item(), img1.size(0))
                # euclidean transformation
                feature_map2, atten2 = model(img5, use_key)
                pred_transform = aet_model4(feature_map1, feature_map2)
                aet_loss = aet_criterion(pred_transform, aff_para[:, :4])
                AET_loss4.update(aet_loss.item(), img1.size(0))
                # Color Contrast brightness sharpeness transformation
                feature_map2, atten2 = model(img6, use_key)
                pred_transform = aet_model5(feature_map1, feature_map2)
                aet_loss = aet_criterion(pred_transform, oper_params)
                AET_loss5.update(aet_loss.item(), img1.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if use_aet:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'AET1 {AET1.val:.4f} ({AET1.avg:.4f})\t'
                      'AET2 {AET2.val:.4f} ({AET2.avg:.4f})\t'
                      'AET3 {AET3.val:.4f} ({AET3.avg:.4f})\t'
                      'AET4 {AET4.val:.4f} ({AET4.avg:.4f})\t'
                      'AET4 {AET5.val:.4f} ({AET5.avg:.4f})\t'
                      'Closs {Closs.val:.4f} ({Closs.avg:.4f})\t'

                      'Top1 {acc.val:.3f} ({acc.avg:.3f})\t'
                      'Top5 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                    'AvgTop1 {acc2.val:.3f} ({acc2.avg:.3f})\t'
                    .format(
                    epoch,
                    batch_idx + 1,
                    len(valloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    AET1=AET_loss1,
                    AET2=AET_loss2,
                    AET3=AET_loss3,
                    AET4=AET_loss4,
                    AET5=AET_loss5,
                    Closs=Closs,
                    acc=accuracies,
                    acc1=top5,acc2=avg_accuracies))

    if use_aet:
        Val_logger.log({
        'epoch': epoch,
        'AET1': AET_loss1.avg,
        'AET2': AET_loss2.avg,
        'AET3': AET_loss3.avg,
        'AET4': AET_loss4.avg,
        'AET5': AET_loss5.avg,
        'Closs': Closs.avg,
        'top1': accuracies.avg,
        'top5': top5.avg,
        'top1avg':avg_accuracies.avg,
          #  'Entropy':Entropy_Loss.avg,
        })
        if writer!=None:
            writer.add_scalars('Data/AET_Loss1', {Type_Label+'_loss': AET_loss1.avg}, epoch)
            writer.add_scalars('Data/AET_Loss2', {Type_Label + '_loss': AET_loss2.avg}, epoch)
            writer.add_scalars('Data/AET_Loss3', {Type_Label + '_loss': AET_loss3.avg}, epoch)
            writer.add_scalars('Data/AET_Loss4', {Type_Label + '_loss': AET_loss4.avg}, epoch)
            writer.add_scalars('Data/AET_Loss5', {Type_Label + '_loss': AET_loss5.avg}, epoch)
            writer.add_scalars('Data/CLoss', {Type_Label+'_loss': Closs.avg}, epoch)
            #writer.add_scalars('Data/Entropy', {Type_Label + '_loss': Entropy_Loss.avg}, epoch)
            writer.add_scalars('Accuracy/top1', {Type_Label+'_accu': accuracies.avg}, epoch)
            writer.add_scalars('Accuracy/top5', {Type_Label+'_accu': top5.avg}, epoch)
            writer.add_scalars('Accuracy/top1avg', {Type_Label + '_accu': avg_accuracies.avg}, epoch)

    return Closs.avg,accuracies.avg