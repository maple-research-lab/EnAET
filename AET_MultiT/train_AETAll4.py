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
import numpy as np
import torchvision.utils as vutils
def Calculate_CLoss(model,mixmatch_criterion,input_list,target_list,Alpha,
                    batch_size,label_batch,loss_iteration,params,mixup_count,writer):
    all_inputs = torch.cat(
        input_list,
        dim=0)
    all_targets = torch.cat(target_list, dim=0)
    lambda_mixup = np.random.beta(Alpha, Alpha)
    lambda_mixup = max(lambda_mixup, 1 - lambda_mixup)
    idx = torch.randperm(all_inputs.size(0))
    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]
    mixed_input = lambda_mixup * input_a + (1 - lambda_mixup) * input_b
    if loss_iteration==0 and writer!=None:
        tmp1 = vutils.make_grid(mixed_input, normalize=True, scale_each=True)
        writer.add_image('Image1_mix', tmp1, loss_iteration)
    mixed_target = lambda_mixup * target_a + (1 - lambda_mixup) * target_b
    # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
    mixed_input = list(torch.split(mixed_input, batch_size))
    mixed_input = interleave(mixed_input, batch_size)  # first 0:batchsize labeled data, others:unlabelled data
    logits = [model(mixed_input[0])]
    for input in mixed_input[1:]:
        logits.append(model(input))
    logits = interleave(logits, batch_size)
    logits_x = torch.cat(logits[0:label_batch], dim=0)
    logits_u = torch.cat(logits[label_batch:], dim=0)
    Lx, Lu, w = mixmatch_criterion(logits_x, mixed_target[:batch_size * label_batch], logits_u, mixed_target[batch_size * label_batch:],
                                   loss_iteration)
    loss = Lx + w * Lu
    loss_ent=0


    return loss,loss_ent,Lx,Lu,w

def Calculate_KLloss(model,img,pred_u):
    pred_T=model(img)
    logp_hat = F.log_softmax(pred_T, dim=1)
    KLloss=F.kl_div(logp_hat, pred_u, size_average=None, reduce=None, reduction='batchmean')
    return KLloss

def Calculate_KLloss_Result(pred_T,pred_u):
    logp_hat = F.log_softmax(pred_T, dim=1)
    KLloss=F.kl_div(logp_hat, pred_u, size_average=None, reduce=None, reduction='batchmean')
    return KLloss
def train_AETAll4(train_dataloader,valid_dataloader, model, optimizer,
                   ema_optimizer, mixmatch_criterion, epoch,use_cuda,
                   aet_model,use_aet,aet_criterion,aet_optimizer,iteration,
                   Temporature,Alpha,lambda_aet,epoch_logger, batch_logger,
                   run_type,writer,num_classes,params,ema_model):
    model.train()
    ema_model.eval()#do not train,only use for validation
    if use_aet:
        aet_model1 = aet_model[0]
        aet_model2 = aet_model[1]
        aet_model3 = aet_model[2]
        aet_model4 = aet_model[3]
        #aet_model5 = aet_model[4]
        aet_model1.train()
        aet_model2.train()
        aet_model3.train()
        aet_model4.train()
        #aet_model5.train()
        aet_optimizer1 = aet_optimizer[0]
        aet_optimizer2 = aet_optimizer[1]
        aet_optimizer3 = aet_optimizer[2]
        aet_optimizer4 = aet_optimizer[3]
        #aet_optimizer5 = aet_optimizer[4]
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    AET_loss1 = AverageMeter()
    AET_loss2 = AverageMeter()
    AET_loss3 = AverageMeter()
    AET_loss4 = AverageMeter()
    AET_loss5 = AverageMeter()
    Closs = AverageMeter()  # loss_x in mixmatch
    Enloss = AverageMeter()  # loss_u in mixmatch
    Entropy_Loss=AverageMeter()
    WS = AverageMeter()
    accuracies = AverageMeter()
    top5 = AverageMeter()
    WAET1 = AverageMeter()
    WAET2 = AverageMeter()
    WAET3 = AverageMeter()
    WAET4 = AverageMeter()
    WAET5 = AverageMeter()
    KL_Loss=AverageMeter()
    TeacherKL = AverageMeter()
    end_time = time.time()
    labeled_train_iter = iter(train_dataloader)
    unlabeled_train_iter = iter(valid_dataloader)
    for batch_idx in range(iteration):
        try:
            (img1, img1_t), img2t, img3t, img4t, img5t, img6t, img7t,aff_parat, transform_matrix1t,oper_paramst, target = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(train_dataloader)
            (img1, img1_t), img2t, img3t, img4t, img5t, img6t, img7t,aff_parat, transform_matrix1t,oper_paramst, target = next(labeled_train_iter)
        #img1 = read_data[0]  # original images
        #target = read_data[1]  # target label

        try:
            (img1_u, img1_u2), img2, img3, img4, img5, img6, img7,aff_para, transform_matrix1,oper_params, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(valid_dataloader)
            (img1_u, img1_u2), img2, img3, img4, img5, img6, img7,aff_para, transform_matrix1,oper_params, _ = unlabeled_train_iter.next()
        if batch_idx == 0 and epoch%10==0:
            if writer != None:
                tmp1 = vutils.make_grid(img1_u, normalize=True, scale_each=True)
                writer.add_image('Image1_u', tmp1, epoch)
                tmp1 = vutils.make_grid(img1_u2, normalize=True, scale_each=True)
                writer.add_image('Image1_u2', tmp1, epoch)
                tmp1 = vutils.make_grid(img2, normalize=True, scale_each=True)
                writer.add_image('Image_Projective', tmp1, epoch)
                tmp1 = vutils.make_grid(img3, normalize=True, scale_each=True)
                writer.add_image('Image_Affine', tmp1, epoch)
                tmp1 = vutils.make_grid(img4, normalize=True, scale_each=True)
                writer.add_image('Image_Similarity', tmp1, epoch)
                tmp1 = vutils.make_grid(img5, normalize=True, scale_each=True)
                writer.add_image('Image_Euclidean', tmp1, epoch)
                tmp1 = vutils.make_grid(img6, normalize=True, scale_each=True)
                writer.add_image('Image_CCBS', tmp1, epoch)
                tmp1 = vutils.make_grid(img7, normalize=True, scale_each=True)
                writer.add_image('Image_Patch', tmp1, epoch)
        data_time.update(time.time() - end_time)
        batch_size = img1.size(0)
        targets_x = torch.zeros(batch_size, num_classes).scatter_(1, target.view(-1, 1), 1)
        if use_cuda:
            img1, targets_x = img1.cuda(), targets_x.cuda(non_blocking=True)
            img1_t=img1_t.cuda()
            target = target.cuda()
            img1_u = img1_u.cuda()
            img1_u2 = img1_u2.cuda()
            img2 = img2.cuda()
            img3 = img3.cuda()
            img4 = img4.cuda()
            img5 = img5.cuda()
            img6 = img6.cuda()
            img7=img7.cuda()
            # img2t = img2t.cuda()
            # img3t = img3t.cuda()
            # img4t = img4t.cuda()
            # img5t = img5t.cuda()
            #img6t = img6t.cuda()
            #img7t = img7t.cuda()
            if use_aet:
                aff_para = aff_para.cuda()
                transform_matrix1 = transform_matrix1.cuda()
                oper_params=oper_params.cuda()
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(img1_u)
            outputs_u2 = model(img1_u2)
            #outputs_u3 = model(img2)
            #outputs_u4 = model(img3)
            #outputs_u5= model(img4)
            #outputs_u6 = model(img5)
            #outputs_u7 = model(img6)
            #outputs_u8 = model(img7)
            if params['mix_mode']==0:
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1))/2#+ torch.softmax(outputs_u3, dim=1)
                # + torch.softmax(outputs_u4, dim=1)+ torch.softmax(outputs_u5, dim=1)+ torch.softmax(outputs_u6, dim=1)
                 #+ torch.softmax(outputs_u7, dim=1)+ torch.softmax(outputs_u8, dim=1)) / 4  # mean the softmax maatrix
            else:
                outputs_u8 = model(img7)
                p=(torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)+torch.softmax(outputs_u8, dim=1))/3
            pt = p ** (1 / Temporature)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()#also serve as p(y|x) with fixed weight(no propagating)
            #now add KL divergence part
            # outputs_x = model(img1)
            # outputs_x2 = model(img1_t)
            # p_x=(torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1))/2
            # ptx = p_x ** (1 / Temporature)
            # targets_xpred = ptx / ptx.sum(dim=1, keepdim=True)
            # targets_xpred = targets_xpred.detach()  # also serve as p(y|x) with fixed weight(no propagating)
        mixup_count=3
        if params['mix_mode']==0:
            loss1,loss_ent,Lx,Lu,w=Calculate_CLoss(model, mixmatch_criterion, [img1,img1_u,img1_u2], [targets_x,targets_u,targets_u], Alpha,
                        batch_size, 1, epoch + batch_idx/iteration, params, mixup_count,writer)
        else:
            loss1, loss_ent, Lx, Lu, w = Calculate_CLoss(model, mixmatch_criterion, [img1, img1_u, img1_u2,img7],
                                                         [targets_x, targets_u, targets_u,targets_u], Alpha,
                                                         batch_size, 1, epoch + batch_idx / iteration, params,
                                                         mixup_count,writer)
            #if batch_idx==0 and epoch==0:
            #    #write the result of mix up to show
        Closs.update(Lx.item(),1*batch_size)
        Enloss.update(Lu.item(),1*batch_size)
        WS.update(w, 1*batch_size)
        loss=loss1#here we do not add any parameter





        if use_aet:
            use_key = ['Attention']
            feature_map1, atten1 = model(img1_u, use_key)
            if params['KL_Lambda']!=0:
                #calculate KL mode
                use_key1 = use_key + ['classifier']
                feature_map2, atten2 = model(img2, use_key1)
                pred_img2=feature_map2[1]
                feature_map2=feature_map2[0]
            else:
                feature_map2, atten2 = model(img2, use_key)
            pred_transform = aet_model1(feature_map1, feature_map2)
            transform_matrix1 = transform_matrix1.view(-1, 8)
            aet_loss = aet_criterion(pred_transform, transform_matrix1)
            loss += aet_loss * min(lambda_aet * np.clip((epoch * iteration + batch_idx) / 1048576.0, 0.0, 1.0),
                                   params['max_lambda'])
            WAET1.update(
                min(lambda_aet * np.clip((epoch * iteration + batch_idx) / 1048576.0, 0.0, 1.0), params['max_lambda']),
                img1.size(0))
            if params['KL_Lambda']!=0:
                #calculate KL mode
                use_key1 = use_key + ['classifier']
                feature_map2, atten2 = model(img3, use_key1)
                pred_img3=feature_map2[1]
                feature_map2=feature_map2[0]
            else:
                feature_map2, atten2 = model(img3, use_key)
            pred_transform2 = aet_model2(feature_map1, feature_map2)
            aet_loss2 = aet_criterion(pred_transform2, aff_para)
            loss += aet_loss2 * min(params['lambda1'] * np.clip((epoch * iteration + batch_idx) / 1048576.0, 0.0, 1.0),
                                    params['max_lambda1'])
            WAET2.update(
                min(params['lambda1'] * np.clip((epoch * iteration + batch_idx) / 1048576.0, 0.0, 1.0),
                    params['max_lambda1']),
                img1.size(0))
            if params['KL_Lambda']!=0:
                #calculate KL mode
                use_key1 = use_key + ['classifier']
                feature_map2, atten2 = model(img4, use_key1)
                pred_img4=feature_map2[1]
                feature_map2=feature_map2[0]
            else:
                feature_map2, atten2 = model(img4, use_key)
            pred_transform3 = aet_model3(feature_map1, feature_map2)
            aet_loss3 = aet_criterion(pred_transform3, aff_para[:, :5])
            loss += aet_loss3 * min(params['lambda2'] * np.clip((epoch * iteration + batch_idx) / 1048576.0, 0.0, 1.0),
                                    params['max_lambda2'])
            WAET3.update(
                min(params['lambda2'] * np.clip((epoch * iteration + batch_idx) / 1048576.0, 0.0, 1.0),
                    params['max_lambda2']),
                img1.size(0))
            if params['KL_Lambda']!=0:
                #calculate KL mode
                use_key1 = use_key + ['classifier']
                feature_map2, atten2 = model(img5, use_key1)
                pred_img5=feature_map2[1]
                feature_map2=feature_map2[0]
            else:
                feature_map2, atten2 = model(img5, use_key)
            pred_transform4 = aet_model4(feature_map1, feature_map2)
            aet_loss4 = aet_criterion(pred_transform4, aff_para[:, :4])
            loss += aet_loss4 * min(params['lambda3'] * np.clip((epoch * iteration + batch_idx) / 1048576.0, 0.0, 1.0),
                                    params['max_lambda3'])
            WAET4.update(
                min(params['lambda3'] * np.clip((epoch * iteration + batch_idx) / 1048576.0, 0.0, 1.0),
                    params['max_lambda3']),
                img1.size(0))

        # Add KL divergence now for all the other augmentations
        if params['KL_Lambda']!=0:
            KLloss = 0
            count_KL = 0
            if params['max_lambda'] != 0:  # designed for ablation study
                KLloss += Calculate_KLloss_Result(pred_img2, targets_u)
                count_KL += 1
            if params['max_lambda1'] != 0:
                KLloss += Calculate_KLloss_Result(pred_img3, targets_u)
                count_KL += 1
            if params['max_lambda2'] != 0:
                KLloss += Calculate_KLloss_Result(pred_img4, targets_u)
                count_KL += 1
            if params['max_lambda3'] != 0:
                KLloss += Calculate_KLloss_Result(pred_img5, targets_u)
                count_KL += 1
            # if params['max_lambda4'] != 0:#here we do not have this for cifar10
            #     KLloss += Calculate_KLloss_Result(pred_img6, targets_u)
            #     count_KL += 1
            if params['mix_mode'] != 0:
                pred_img7 = model(img7)  # again used for the final KL loss in agreement with the Teacher model
                KLloss += Calculate_KLloss_Result(pred_img7, targets_u)
                count_KL += 1
            if count_KL != 0:
                KLloss = KLloss / count_KL
        if params['KL_Lambda']!=0:

            loss+=KLloss*params['KL_Lambda']
            KL_Loss.update(KLloss.item(),batch_size)
        losses.update(loss.item(), img1.size(0))
        if use_aet:
            AET_loss1.update(aet_loss.item(), img1.size(0))
            AET_loss2.update(aet_loss2.item(), img1.size(0))
            AET_loss3.update(aet_loss3.item(), img1.size(0))
            AET_loss4.update(aet_loss4.item(), img1.size(0))
            #AET_loss5.update(aet_loss5.item(), img1.size(0))
            aet_optimizer1.zero_grad()
            aet_optimizer2.zero_grad()
            aet_optimizer3.zero_grad()
            aet_optimizer4.zero_grad()
            #aet_optimizer5.zero_grad()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if params['use_ema']:
            ema_optimizer.step()
        if use_aet:
            aet_optimizer1.step()
            aet_optimizer2.step()
            aet_optimizer3.step()
            aet_optimizer4.step()
            #aet_optimizer5.step()
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        with torch.no_grad():#save gpu space
            classify_output = model(img1)
        prec1, prec5 = Calculate_top_accuracy(classify_output.data, target.data, topk=(1, 5))
        accuracies.update(prec1.item(), img1.size(0))
        top5.update(prec5.item(), img1.size(0))
        if use_aet:
            batch_logger.log({
                'epoch': epoch,
                'batch': batch_idx + 1,
                'iter': (epoch) * iteration + (batch_idx + 1),
                'loss': losses.val,
                'AET1': AET_loss1.val,
                'AET2': AET_loss2.val,
                'AET3': AET_loss3.val,
                'AET4': AET_loss4.val,
                'AET5': AET_loss5.val,
                'Closs': Closs.val,
                'Eloss': Enloss.val,
                'KL':KL_Loss.val,
                'wd': WS.val,
                'waet': WAET1.val,
                'top1': accuracies.val,
                'top5': top5.val,
                'lr1': optimizer.param_groups[0]['lr'],
                'lr2': aet_optimizer1.param_groups[0]['lr']
            })
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'AET1 {AET1.val:.4f} ({AET1.avg:.4f})\t'
              'AET2 {AET2.val:.4f} ({AET2.avg:.4f})\t'
              'AET3 {AET3.val:.4f} ({AET3.avg:.4f})\t'
              'AET4 {AET4.val:.4f} ({AET4.avg:.4f})\t'
              'AET5 {AET5.val:.4f} ({AET5.avg:.4f})\t'
              'Closs {Closs.val:.4f} ({Closs.avg:.4f})\t'
              'Enloss {Enloss.val:.4f} ({Enloss.avg:.4f})\t'
              'KL {KL.val:.4f} ({KL.avg:.4f})\t'
              'Top1 {acc.val:.3f} ({acc.avg:.3f})\t'
              'Top5 {acc1.val:.3f} ({acc1.avg:.3f})\t'.format(
            epoch,
            batch_idx + 1,
            iteration,
            batch_time=batch_time,
            data_time=data_time,
            AET1=AET_loss1, AET2=AET_loss2, AET3=AET_loss3, AET4=AET_loss4,AET5=AET_loss5,
            Closs=Closs,
            Enloss=Enloss,
            KL=KL_Loss,
            loss=losses,
            acc=accuracies,
            acc1=top5))
    if use_aet:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'AET1': AET_loss1.avg,
            'AET2': AET_loss2.avg,
            'AET3': AET_loss3.avg,
            'AET4': AET_loss4.avg,
            'AET5': AET_loss5.avg,
            'Closs': Closs.avg,
            'Eloss': Enloss.avg,
            'KL':KL_Loss.avg,
            'wd': WS.avg,
            'waet': WAET1.avg,
            'top1': accuracies.avg,
            'top5': top5.avg,
            'lr1': optimizer.param_groups[0]['lr'],
            'lr2': aet_optimizer1.param_groups[0]['lr']
        })
        if writer != None:
            writer.add_scalars('Data/Loss', {'train_loss': losses.avg}, epoch)
            writer.add_scalars('Data/AET_Loss1', {'train_loss': AET_loss1.avg}, epoch)
            writer.add_scalars('Data/AET_Loss2', {'train_loss': AET_loss2.avg}, epoch)
            writer.add_scalars('Data/AET_Loss3', {'train_loss': AET_loss3.avg}, epoch)
            writer.add_scalars('Data/AET_Loss4', {'train_loss': AET_loss4.avg}, epoch)
            writer.add_scalars('Data/AET_Loss5', {'train_loss': AET_loss5.avg}, epoch)
            writer.add_scalars('Data/CLoss', {'train_loss': Closs.avg}, epoch)
            writer.add_scalars('Data/ELoss', {'train_loss': Enloss.avg}, epoch)
           # writer.add_scalars('Data/Entropy', {'train_loss': Entropy_Loss.avg}, epoch)
            writer.add_scalars('Data/KLloss', {'train_loss': KL_Loss.avg}, epoch)
            #writer.add_scalars('Data/KL_Teacher_loss', {'train_loss': TeacherKL.avg}, epoch)
            writer.add_scalars('Accuracy/top1', {'train_accu': accuracies.avg}, epoch)
            writer.add_scalars('Accuracy/top5', {'train_accu': top5.avg}, epoch)
            writer.add_scalar('Weight/Lambda1', WAET1.avg, epoch)
            writer.add_scalar('Weight/Lambda2', WAET2.avg, epoch)
            writer.add_scalar('Weight/Lambda3', WAET3.avg, epoch)
            writer.add_scalar('Weight/Lambda4', WAET4.avg, epoch)
            writer.add_scalar('Weight/Lambda5', WAET5.avg, epoch)
            writer.add_scalar('Weight/Beta', WS.avg, epoch)
            writer.add_scalar('LR/LR1', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('LR/LR2', aet_optimizer1.param_groups[0]['lr'], epoch)
            if epoch % 10 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram('CLF' + name, param.clone().cpu().data.numpy(), epoch)
                for name, param in aet_model1.named_parameters():
                    writer.add_histogram('AET1' + name, param.clone().cpu().data.numpy(), epoch)
                for name, param in aet_model2.named_parameters():
                    writer.add_histogram('AET2' + name, param.clone().cpu().data.numpy(), epoch)
                for name, param in aet_model3.named_parameters():
                    writer.add_histogram('AET3' + name, param.clone().cpu().data.numpy(), epoch)
                for name, param in aet_model4.named_parameters():
                    writer.add_histogram('AET4' + name, param.clone().cpu().data.numpy(), epoch)
                # for name, param in aet_model5.named_parameters():
                #     writer.add_histogram('AET5' + name, param.clone().cpu().data.numpy(), epoch)
    if params['use_ema']:
        ema_optimizer.step(bn=True)
    return losses.avg, Closs.avg, Enloss.avg










def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets
def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]