# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
def add_noBiasWeightDecay(model, skip_list):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    assert len(list(model.parameters())) == (len(decay) + len(no_decay))

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}]

def add_Connector_nobiasweightdecay(model,skip_list):
    decay,no_decay=[],[]
    clf_decay, clf_no_decay = [], []
    classify_params = list(map(id, model.module.clf.parameters()))
    model_param = list(map(id, model.parameters()))
    print('Previously we have %d parameters' % len(model_param))
    print('in classifier we have %d parameters' % len(classify_params))
    count1=0
    count2=0
    count11 = 0
    count21 = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weight
        tmp_id=id(param)
        if tmp_id not in classify_params:
            count1+=1
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
                count11+=1
            else:
                decay.append(param)
        else:
            count2+=1
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                clf_no_decay.append(param)
                count21+=1
            else:
                clf_decay.append(param)
    print('%d parameter in non-classifier part, %d parameter in classifier part'%(count1,count2))
    print('%d no weight decay parameter in non-classifier part, %d no weight decay parameter in classifier part' % (count11, count21))
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}],\
            [{'params': clf_no_decay, 'weight_decay': 0.0}, {'params': clf_decay}]