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

class STL10(object):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

        Args:
            root (string): Root directory of dataset where directory
                ``stl10_binary`` exists.
            split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
                Accordingly dataset is selected.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.

    """
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')
    def __init__(self, root):
        self.root=os.path.expanduser(root)
        self.final_path = os.path.join(self.root, 'STL10')
        mkdir(self.final_path)
        self.train_path = os.path.join(self.final_path, 'trainset')
        self.test_path = os.path.join(self.final_path, 'testset')
        self.extra_path = os.path.join(self.final_path, 'unlabelset')
        mkdir(self.train_path)
        mkdir(self.test_path)
        mkdir(self.extra_path)
        if not self._check_integrity():
            self.download()
        check_path=os.path.join(self.train_path,'trainset.npy')
        if not os.path.exists(check_path)or os.path.getsize(check_path) < 10000:
            self.Process_Dataset(self.train_path,'train')
        check_path = os.path.join(self.test_path, 'trainset.npy')
        if not os.path.exists(check_path)or os.path.getsize(check_path) < 10000:
            self.Process_Dataset(self.test_path,'test')
        check_path = os.path.join(self.extra_path, 'trainset.npy')
        if not os.path.exists(check_path)or os.path.getsize(check_path) < 10000:
            self.Process_Dataset(self.extra_path,'extra')
    def Process_Dataset(self,train_path,split):
        if split=='train':
            data_path=self.train_list[0][0]
            label_path=self.train_list[1][0]
        elif split=='test':
            data_path = self.test_list[0][0]
            label_path = self.test_list[1][0]
        elif split=='extra':
            data_path=self.train_list[2][0]
            label_path=None

        labels = None
        if label_path:
            path_to_labels = os.path.join(
                self.root, self.base_folder, label_path)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_path)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        if label_path==None:
            labels=np.zeros(len(images),dtype=np.uint8)-1#all -1 denotes no labels
        #started to save as a file
        data_save_path=os.path.join(train_path,'trainset.npy')
        label_save_path=os.path.join(train_path,'aimset.npy')
        #return images, labels
        np.save(data_save_path,images)
        np.save(label_save_path,labels)


    def _check_integrity(self,):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)




