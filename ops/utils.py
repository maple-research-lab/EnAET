import csv
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

import numpy as np
def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    #
    # _, pred = outputs.topk(1, 1, True)
    # pred = pred.t()
    # correct = pred.eq(targets.view(1, -1))
    # n_correct_elems = correct.float().sum().data[0]
    # n_correct_elems=0
    # classes=4
    # for i in range(batch_size):
    #     correct_flag=True
    #     max_index=0
    #     max_possibility=0
    #     for j in range(classes):
    #         if outputs[i][j]>max_possibility:
    #             max_index=j+1
    #             max_possibility=outputs[i][j]
    #     if max_index==targets[i]:
    #         n_correct_elems+=1
    #correct = (targets.eq(outputs.long())).sum()
    #n_correct_elems = 0

    # _, predicted = torch.max(outputs.data, 1)
    # batch_size= targets.size(0)
    # correct=0
    # correct += (predicted == targets).sum()
    _, predicted = torch.max(outputs, 1)
    nb_corrects = float((predicted == targets).sum().data)
    batch_accuracy = nb_corrects / batch_size
    return batch_accuracy
def Calculate_top_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res