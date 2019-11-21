# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
from ops.os_operation import mkdir
import os
from torchvision.datasets.utils import download_url, check_integrity
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import numpy as np
import scipy.io as sio
class SVHN(object):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root):
        self.root = root
        self.final_path = os.path.join(self.root, 'SVHN')
        mkdir(self.final_path)
        self.train_path = os.path.join(self.final_path, 'trainset')
        self.test_path = os.path.join(self.final_path, 'testset')
        self.extra_path=os.path.join(self.final_path,'extraset')
        mkdir(self.train_path)
        mkdir(self.test_path)
        mkdir(self.extra_path)
        if os.path.getsize(self.train_path) < 10000:
            self.Process_Dataset(self.train_path,'train')
        if os.path.getsize(self.test_path) < 10000:
            self.Process_Dataset(self.test_path,'test')
        if os.path.getsize(self.extra_path) < 10000:
            self.Process_Dataset(self.extra_path,'extra')

    def Process_Dataset(self, train_path, split):
        url = self.split_list[split][0]
        filename = self.split_list[split][1]
        file_md5 = self.split_list[split][2]
        self.download(url,filename,file_md5)
        if not self._check_integrity(file_md5,filename):
            self.download(url,filename,file_md5)#download again
        loaded_mat = sio.loadmat(os.path.join(self.root, filename))
        data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(labels, labels == 10, 0)
        data = np.transpose(data, (3, 2, 0, 1))
        #in order that i do not need to rewrite dataloader again, I changed the way to processing it.
        # data_path=os.path.join(train_path,'trainset.npy')
        # aim_path=os.path.join(train_path,'aimset.npy')
        # np.save(data_path,data)
        # np.save(aim_path,labels)
        for i in range(len(data)):
            tmp_train_path=os.path.join(train_path,'trainset'+str(i)+'.npy')
            tmp_aim_path = os.path.join(train_path, 'aimset' + str(i) + '.npy')
            np.save(tmp_train_path,data[i])
            np.save(tmp_aim_path,labels[i])
    def download(self,url,filename,md5):

        download_url(url, self.root, filename, md5)

    def _check_integrity(self,file_md5,filename):
        root = self.root
        fpath = os.path.join(root, filename)
        return check_integrity(fpath, file_md5)

