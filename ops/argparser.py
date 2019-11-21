# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/

import parser
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F',type=str, required=True,help='training data path')#File path for our MAINMAST code
    parser.add_argument('-M', type=str,default='Train_Model', help='model path for evluation and the path to save model')
    parser.add_argument('--mode',  type=int, required=True, help='0:default mode to run\n')
    parser.add_argument('--epochs', default=1024, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                     help='initial learning rate')
    parser.add_argument('--lr1',  default=0.1, type=float,
                        help='initial learning rate for new optimizer1')
    parser.add_argument('--reg',  default=1e-7, type=float,
                        help='initial l2 regularization lambda')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--patience', default=10, type=int,
                        help='patient epoch for updating the lr scheduler')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--seed', type=int,default=888, help='manual seed')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--choose', type=str, default='0', help='specified gpu')
    parser.add_argument('--num_workers', type=int, default=16, help='number of data loading workers')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--rot', type=float, default=180, help='range of angle for rotation, default:[-180, 180]')
    parser.add_argument('--shear', type=float, default=30, help='range of angle for shearing, default: [-30, 30]')
    parser.add_argument('--translate', type=float, default=0.2,
                        help='range of ratio of translation to the height/width of the image')
    parser.add_argument('--shrink', type=float, default=0.8, help='the lower bound of scaling')
    parser.add_argument('--enlarge', type=float, default=1.2, help='the higher bound of scaling')
    parser.add_argument('--shift',type=float,default=4,help='shift parameter for projective data changing method')
    parser.add_argument('--log_path', type=str, default='train_log', help='training log record path')
    parser.add_argument('--model_path',type=str,default='train_model',help="training result record path")
    parser.add_argument('--type',type=int,default=0,help='specify the model\n'
                                                         '0: Wide ResNet-28-2\n'
                        '1: Wide ResNet-28-2-Large')

    parser.add_argument('--resume',type=int,default=0,help='reload trained model to continue training')
    parser.add_argument('--KL_Lambda', default=1.0, type=float, help='hyper parameter to control the KL divergence loss term')
    parser.add_argument('--lambda',type=float,default=10,help='warm factor in combined loss of projective AET and classifier loss')
    parser.add_argument('--lambda1', type=float, default=7.5,
                        help='warm factor in combined loss of affine AET and classifier loss')
    parser.add_argument('--lambda2', type=float, default=5,
                        help='warm factor in combined loss of similarity AET and classifier loss')
    parser.add_argument('--lambda3', type=float, default=2,
                        help='warm factor in combined loss of euclidean AET and classifier loss')
    parser.add_argument('--lambda4', type=float, default=0.5,
                        help='warm factor in combined loss ofCCBS AET and classifier loss')
    parser.add_argument('--max_lambda', type=float, default=1.0,
                        help='balanced factor in combined loss of AET and classifier loss')
    parser.add_argument('--max_lambda1', type=float, default=0.75,
                        help='balanced factor in combined loss of affine AET and classifier loss')
    parser.add_argument('--max_lambda2', type=float, default=0.5,
                        help='balanced factor in combined loss of similarity AET and classifier loss')
    parser.add_argument('--max_lambda3', type=float, default=0.2,
                        help='balanced factor in combined loss of euclidean AET and classifier loss')
    parser.add_argument('--max_lambda4', type=float, default=0.05,
                        help='balanced factor in combined loss of CCBS AET and classifier loss')
    parser.add_argument('--portion',type=float,default=0.08,help='percentage of data with labels')
    parser.add_argument('--beta', type=float, default=75,help='hyper parameter for the consistency loss in MixMatch part')
    parser.add_argument('--ema_decay', default=0.999, type=float,help='EMA decay hyper-parameter')
    parser.add_argument('--T', default=0.5, type=float,help='Temperature settings applied for the sharpening')
    parser.add_argument('--alpha', default=0.75, type=float,help='Alpha settings for the mixup part')
    parser.add_argument('--mix_iteration', default=1024, type=int, help='Required iteration for mixmatch in each epoch')
    parser.add_argument('--start_epoch', default=0, type=int, help='startring epoch for resuming situation')
    parser.add_argument('--dataset',default='cifar10',type=str,help='Choose dataset for training')
    parser.add_argument('--tensorboard', default=1, type=int, help='Use tensorboard to keep results or not. Default: True')
    parser.add_argument('--mix_mode',default=0,type=int,help='Mix up mode: 0:choose default mixmatch mixup mode\n 1: choose mixup with mosaic operations')
    parser.add_argument('--Mixmatch_warm',default=16,type=int,help='Steps that necessary for warming up the enloss term in Mixmatch')
    parser.add_argument('-S',default='',type=str,help='the path to save all the training logs and models')
    parser.add_argument('--use_ema',default=1,type=int,help='Use ema or not during training:default=1')
    args = parser.parse_args()
    params = vars(args)
    return params