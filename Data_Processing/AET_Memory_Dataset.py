import torch
import torch.utils.data as data
import numpy as np
import random
import os
from PIL import Image, PILLOW_VERSION, ImageEnhance
import numbers
from torchvision.transforms.functional import _get_inverse_affine_matrix
import math
from sklearn.model_selection import train_test_split
from collections import defaultdict

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class AET_Memory_Dataloader(data.Dataset):
    def __init__(self, dataset_dir,degrees,dataset_mean,dataset_std,translate=None,shear=None,
                 shift=6, train_label=True, scale=None,  resample=False,
                 fillcolor=0,matrix_transform=None,
                 transform_pre=None, transform=None, target_transform=None, rand_state=888,
                 valid_size=0.1,uniform_label=False,num_classes=10,patch_length=8,extra_path=None):
        super(AET_Memory_Dataloader, self).__init__()
        self.root=os.path.abspath(dataset_dir)
        self.dataset_mean=dataset_mean
        self.dataset_std=dataset_std
        self.shift=shift
        self.trainsetFile = []
        self.aimsetFile = []
        self.trainlist = os.path.join(dataset_dir, 'trainset.npy')
        self.aimlist = os.path.join(dataset_dir,'aimset.npy')
        self.trainlist=np.load(self.trainlist)
        self.aimlist=np.load(self.aimlist)
        if extra_path!=None:
            self.trainsetFile=os.path.join(extra_path,'trainset.npy')
            self.aimsetFile=os.path.join(extra_path,'aimset.npy')
            self.trainsetFile=np.load(self.trainsetFile)
            self.aimsetFile=np.load(self.aimsetFile)
            print("We add %d unlabeled data for training" % len(self.aimsetFile))
            self.trainlist=np.concatenate((self.trainlist,self.trainsetFile),axis=0)
            self.aimlist=np.concatenate((self.aimlist,self.aimsetFile),axis=0)
        self.train_label=train_label
        self.valid_size=valid_size
        #self.unlabel_Data=unlabel_Data
        # here update this with 80% as training, 20%as validation
        if valid_size>0:
            if uniform_label==False:

                X_train, X_test, y_train, y_test = train_test_split(self.trainlist, self.aimlist, test_size=valid_size,
                                                            random_state=rand_state)
                if train_label:
                    self.trainlist = X_train
                    self.aimlist = y_train

                else:
                    self.trainlist = X_test
                    self.aimlist = y_test
            else:
                #pick the uniform valid size indicated
                shuffle_range=np.arange(len(self.trainlist))
                random.seed(rand_state)
                random.shuffle(shuffle_range)
                require_size=int(len(self.aimlist)*valid_size/num_classes)
                self.trainlist,self.aimlist=self.pick_top_k_example(require_size,shuffle_range,num_classes)
                print('After Picking, now we have %d trainlist '%len(self.trainlist))
                print(self.trainlist.shape)
        if uniform_label==True and len(self.trainlist)<50000:
            #to accelerate training to avoid dataloader load again and again for small data
            repeat_times=int(50000/len(self.trainlist))
            self.trainlist=np.repeat(self.trainlist,repeat_times,axis=0)
            self.aimlist=np.repeat(self.aimlist,repeat_times,axis=0)
            print('After repeating, now we have %d trainlist ' % len(self.trainlist))
            print(self.trainlist.shape)
        self.aimlist=np.array(self.aimlist,dtype=np.int64)
        self.transform_pre = transform_pre
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale
        self.resample = resample
        self.fillcolor = fillcolor
        self.transform = transform
        self.target_transform = target_transform
        self.matrix_transform=matrix_transform
        self.degrees = degrees
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate
        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.patch_length=patch_length
    def pick_top_k_example(self,img_per_cat,shuffle_range,num_class):
        record_dict=defaultdict(list)
        for i in range(len(shuffle_range)):
            tmp_id=shuffle_range[i]
            label=int(self.aimlist[tmp_id])
            if label not in record_dict:
                record_dict[label].append(tmp_id)
            elif len(record_dict[label])<img_per_cat:
                record_dict[label].append(tmp_id)
            break_flag=True
            if len(record_dict)<num_class:
                break_flag=False
            for tmp_label in record_dict.keys():
                tmp_length=len(record_dict[tmp_label])
                if tmp_length<img_per_cat:
                    break_flag=False
                    break
            if break_flag:
                break
        #specify new trainlist and aimlist
        assert len(record_dict)==num_class
        for tmp_label in record_dict.keys():
            tmp_length = len(record_dict[tmp_label])
            assert tmp_length==img_per_cat
        train_list=[]
        aim_list=[]
        for tmp_label in record_dict.keys():
            tmp_list=record_dict[tmp_label]
            for tmp_id in tmp_list:
                train_list.append(self.trainlist[tmp_id])
                aim_list.append(self.aimlist[tmp_id])
        train_list=np.array(train_list)
        aim_list=np.array(aim_list)
        return train_list,aim_list



    @staticmethod
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        # random generate angle,translate, scale and shear in a range
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear
    def normalise(self,x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) ):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x -= mean * 255
        x *= 1.0 / (255 * std)
        return x
    def denormalize(self,x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x=x*(255*std)
        x+=mean*255
        return x

    def operate_CCBS(self,img1_transform):
        oper_param=np.random.uniform(low=0.2,high=1.8,size=4).astype('float32')
        image1 = ImageEnhance.Color(img1_transform).enhance(oper_param[0])
        image1 = ImageEnhance.Contrast(image1).enhance(oper_param[1])
        image1 = ImageEnhance.Brightness(image1).enhance(oper_param[2])
        image1 = ImageEnhance.Sharpness(image1).enhance(oper_param[3])
        return image1,oper_param
    def cut_out(self,img):
        """
                Args:
                    img (Tensor): Tensor image of size (C, H, W).
                Returns:
                    Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.patch_length // 2, 0, h)
        y2 = np.clip(y + self.patch_length // 2, 0, h)
        x1 = np.clip(x - self.patch_length // 2, 0, w)
        x2 = np.clip(x + self.patch_length // 2, 0, w)
        for k in range(img.shape[0]):
            img[k,y1:y2,x1:x2]=np.mean(img[k,y1:y2,x1:x2])
        return img
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        train_path = self.trainlist[index]
        aim_path = self.aimlist[index]
        img1 = train_path#saved in memory,directly load
        target = aim_path

        #if self.unlabel_Data:

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #print (img1.shape)
        #img1_transform = img1.copy()

        img1 = img1.transpose((1, 2, 0))#change 3*32*32 to 32*32*3

        #print(img1.shape)
        #print(img1)
        img1 = self.normalise(img1,self.dataset_mean,self.dataset_std)#normalize#channel last format
        #print(img1.shape)
        #print(img1)
        #exit()
        img1 = img1.transpose((2, 0, 1))
        #change back to 3*32*32
        #print(img1.shape)
        #exit()


        if self.transform_pre is not None:
            #operation should be channel first
            #if self.unlabel_Data:#remove unlabel data label to save a dataloader to free computer cpu usage
            self.transform_now=TransformTwice(self.transform_pre)
            img1,img1_another=self.transform_now(img1)
        # if self.train_label == False and self.valid_size != 0:  # for the labeled dataset
        #     if self.transform is not None:
        #         img1 = self.transform(img1)
        #     if self.target_transform is not None:
        #         target = self.target_transform(target)
        #     img1 = torch.from_numpy(img1)
        #     return img1,target
            #else:
            #img1 = self.transform_pre(img1)
        #print(img1)
        #exit()
        #if self.transform_pre is not None:
        #    img1_transform=self.transform_pre(img1_transform)#get img1 from the tranform result
        #revoke bake img1 to make
        #print(img1)
        #print(img1.shape)
        img1_transform = img1.transpose((1, 2, 0))
        #print(img1_transform.shape)
        img1_transform=self.denormalize(img1_transform,self.dataset_mean,self.dataset_std)
        #print(img1_transform)
        #print(img1_transform.shape)
        #img1_transform = img1.transpose((2,0,1))
        #exit()
        #print(img1.shape())
        #exit()
        img1_transform = Image.fromarray(img1_transform.astype(np.uint8))
        # projective transformation on image2
        width, height = img1_transform.size
        center = (img1_transform.size[0] * 0.5 + 0.5, img1_transform.size[1] * 0.5 + 0.5)
        shift = [float(random.randint(-int(self.shift), int(self.shift))) for ii in range(8)]
        scale = random.uniform(self.scale[0], self.scale[1])
        rotation = random.randint(0, 3)

        pts = [((0 - center[0]) * scale + center[0], (0 - center[1]) * scale + center[1]),
               ((width - center[0]) * scale + center[0], (0 - center[1]) * scale + center[1]),
               ((width - center[0]) * scale + center[0], (height - center[1]) * scale + center[1]),
               ((0 - center[0]) * scale + center[0], (height - center[1]) * scale + center[1])]
        pts = [pts[(ii + rotation) % 4] for ii in range(4)]
        pts = [(pts[ii][0] + shift[2 * ii], pts[ii][1] + shift[2 * ii + 1]) for ii in range(4)]

        coeffs = self.find_coeffs(
            pts,
            [(0, 0), (width, 0), (width, height), (0, height)]
        )

        kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
        img2 = img1_transform.transform((width, height), Image.PERSPECTIVE, coeffs, self.resample, **kwargs)
        #img1_transform = np.array(img1_transform).astype('float32')
        img2 = np.array(img2).astype('float32')
        img2 = self.normalise(img2,self.dataset_mean,self.dataset_std)
        img2 = img2.transpose((2, 0, 1))
        #apply affine transformation here
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img1_transform.size)
        output_size = img1_transform.size  # 32*32
        center = (img1_transform.size[0] * 0.5 + 0.5, img1_transform.size[1] * 0.5 + 0.5)
        matrix = _get_inverse_affine_matrix(center, *ret)
        kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
        img3 = img1_transform.transform(output_size, Image.AFFINE, matrix, self.resample, **kwargs)
        img3= np.array(img3).astype('float32')
        img3 = self.normalise(img3,self.dataset_mean,self.dataset_std)
        img3 = img3.transpose((2, 0, 1))
        aff_para = [math.cos(math.radians(ret[0])),  # degree cos
                    math.sin(math.radians(ret[0])),  # degree sin
                    ret[1][0] / self.translate[0] / output_size[0],  # translate x
                    ret[1][1] / self.translate[1] / output_size[1],  # translate y
                    ret[2] * 2. / (self.scale[1] - self.scale[0]) - (self.scale[0] + self.scale[1]) / (
                            self.scale[1] - self.scale[0]),  # scale
                    ret[3] * 2. / (self.shear[1] - self.shear[0]) - (self.shear[0] + self.shear[1]) / (
                            self.shear[1] - self.shear[0])]  # shear

        aff_para = torch.from_numpy(np.array(aff_para, np.float32, copy=False))  # affine transform parameter
        #apply similarity transformation
        matrix = _get_inverse_affine_matrix(center, ret[0],ret[1],ret[2],0)
        kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
        img4 = img1_transform.transform(output_size, Image.AFFINE, matrix, self.resample, **kwargs)
        img4 = np.array(img4).astype('float32')
        img4 = self.normalise(img4,self.dataset_mean,self.dataset_std)
        img4 = img4.transpose((2, 0, 1))
        #apply eculidean transfomration
        matrix = _get_inverse_affine_matrix(center, ret[0], ret[1], 1.0, 0)
        kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
        img5 = img1_transform.transform(output_size, Image.AFFINE, matrix, self.resample, **kwargs)
        img5 = np.array(img5).astype('float32')
        img5 = self.normalise(img5,self.dataset_mean,self.dataset_std)
        img5 = img5.transpose((2, 0, 1))
        #apply the colorize, contrast, brightness, sharpeness to the image
        img6,oper_params=self.operate_CCBS(img1_transform)
        img6 = np.array(img6).astype('float32')
        img6 = self.normalise(img6,self.dataset_mean,self.dataset_std)
        img6 = img6.transpose((2, 0, 1))
        #add another image with cutout

        img7 = np.array(img1_transform).astype('float32')
        img7 = self.normalise(img7,self.dataset_mean,self.dataset_std)
        img7 = img7.transpose((2, 0, 1))
        img7 = self.cut_out(img7)
        img1_transform = np.array(img1_transform).astype('float32')
        if self.transform is not None:
            img1 = self.transform(img1)
            img1_transform = self.transform(img1_transform)
            img2 = self.transform(img2)
            img3=self.transform(img3)
            img4 = self.transform(img4)
            img5 = self.transform(img5)
            img6 = self.transform(img6)
            img7 = self.transform(img7)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        img3 = torch.from_numpy(img3)
        img4 = torch.from_numpy(img4)
        img5 = torch.from_numpy(img5)
        img6 = torch.from_numpy(img6)
        img7 = torch.from_numpy(img7)
        img1_transform=torch.from_numpy(img1_transform)
        coeffs = torch.from_numpy(np.array(coeffs, np.float32, copy=False)).view(8, 1, 1)
        oper_params=torch.from_numpy(oper_params)
        if self.matrix_transform is not None:
            coeffs = self.matrix_transform(coeffs)
        #if self.unlabel_Data:
            #img1_another = np.array(img1_another).astype('float32')
            #img1_another = self.normalise(img1_another)
            #img1_another = img1_another.transpose((2, 0, 1))
        if self.transform is not None:
            img1_another = self.transform(img1_another)
        if self.transform_pre is not None:
            img1_another = torch.from_numpy(img1_another)
            #print(img1)
            #print(img1_another)
            #exit()
            return (img1,img1_another),img2,img3,img4,img5,img6,img7,aff_para,coeffs,oper_params, target
        else:
            return (img1, img1), img2, img3, img4, img5,img6, img7,aff_para, coeffs,oper_params, target
        #else:
        #    return img1,(img1,img2), (img1,img3),(img1,img4),(img1,img5),aff_para,coeffs, target

    def __len__(self):
        return len(self.aimlist)