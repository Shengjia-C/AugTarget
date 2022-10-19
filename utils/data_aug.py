import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import os
import os.path as osp
import sys
import random
import torchvision as t
import numpy as np
import math

__all__ = ['SirstAugDataset_AUG', 'MDFADataset_AUG', 'MergedDataset_AUG']


class MergedDataset_AUG(Data.Dataset):
    def __init__(self, mdfa_base_dir='/home/chenshengjia/data/MDvsFA_cGAN/data', sirstaug_base_dir='/home/chenshengjia/data/sirst_aug', mode='train', base_size=256):
        assert mode in ['train', 'test']

        self.sirstaug = SirstAugDataset_AUG(base_dir=sirstaug_base_dir, mode=mode)
        self.mdfa = MDFADataset_AUG(base_dir=mdfa_base_dir, mode=mode, base_size=base_size)
        
    def __getitem__(self, i):
        if i < self.mdfa.__len__():
            return self.mdfa.__getitem__(i)
        else:
            inx = i - self.mdfa.__len__()
            return self.sirstaug.__getitem__(inx)

    def __len__(self):
        return self.sirstaug.__len__() + self.mdfa.__len__()


class MDFADataset_AUG(Data.Dataset):
    def __init__(self, base_dir='/home/chenshengjia/data/MDvsFA_cGAN/data', mode='train', base_size=256):
    #def __init__(self, base_dir='../data/MDFA', mode='train', base_size=256):
        
        assert mode in ['train', 'test']

        self.mode = mode
        if mode == 'train':
            self.img_dir = osp.join(base_dir, 'training')
            self.mask_dir = osp.join(base_dir, 'training')
        elif mode == 'test':
            self.img_dir = osp.join(base_dir, 'test_org')
            self.mask_dir = osp.join(base_dir, 'test_gt')
        else:
            raise NotImplementedError

        self.img_transform = transforms.Compose([
            transforms.Resize((base_size, base_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((base_size, base_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        
        self.TA2 = Target_Augmentation2()

    def __getitem__(self, i):
        if self.mode == 'train':
            img_path = osp.join(self.img_dir, '%06d_1.png' % i)
            mask_path = osp.join(self.mask_dir, '%06d_2.png' % i)
        elif self.mode == 'test':
            img_path = osp.join(self.img_dir, '%05d.png' % i)
            mask_path = osp.join(self.mask_dir, '%05d.png' % i)
        else:
            raise NotImplementedError

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        img, mask = self.TA2(img, mask)
        
        img = t.transforms.ToPILImage()(img) 
        mask = t.transforms.ToPILImage()(mask)
        img, mask = self.img_transform(img), self.mask_transform(mask)
        return img, mask

    def __len__(self):
        if self.mode == 'train':
            return 9978
        elif self.mode == 'test':
            return 100
        else:
            raise NotImplementedError         

class SirstAugDataset_AUG(Data.Dataset):
    def __init__(self, base_dir='/home/chenshengjia/data/sirst_aug', mode='train'):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])
        self.TA1 = Target_Augmentation1()
        

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        
        img, mask = self.TA1(img, mask)

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        
        return img, mask

    def __len__(self):
        return len(self.names)


    
    


class Target_Augmentation1(object):
   
    # Based on Random Erasing
    
    def __init__(self,  sl = 0.02, sh = 0.4, r1 = 0.3, numbers = 3): 
        
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.target_numbers = numbers
        
    def __call__(self, img, mask):

        area = 10 * 10
 
        w = []
        h = []
        
        target_area = random.uniform(self.sl, self.sh) * area  
        aspect_ratio = random.uniform(self.r1, 1/self.r1)
        
        for i in range(self.target_numbers):
            
            h.append(int(round(math.sqrt(target_area * aspect_ratio))))
            w.append(int(round(math.sqrt(target_area / aspect_ratio)))) 
            
        img_aug = np.array(img)    
        mask_aug = np.array(mask)
        
        img_aug = img_aug.swapaxes(0,2)
        mask_aug = mask_aug.swapaxes(0,1)

        x = []
        y = []
        
        if img_aug.shape[0]== 3:
            
            for i in range(self.target_numbers):

                if w[i] < img_aug.shape[1] and h[i] < img_aug.shape[2]:
                    
                    x.append(random.randint(0, img_aug.shape[1] - h[i])) 
                    y.append(random.randint(0, img_aug.shape[2] - w[i]))
                
                if img_aug[0,x[i],y[i]] > 0:
                            
                        m1, m2 = np.mgrid[:h[i], :w[i]]
                        circle = (m2 - h[i]//2) ** 2 + (m1 - w[i]//2) ** 2
                        
                        target = -target + np.mean((img_aug)[0])
                        target[target < np.min((img_aug)[0])] = np.min((img_aug)[0])
                        theta = int(np.min((img_aug)[0])/np.max((img_aug)[0]) * np.mean((img_aug)[0]))
                        target[target >= theta] = np.max((img_aug)[0])

                        hh = target.shape[0]
                        ww = target.shape[1]
                        
                        img_aug[0, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
                        img_aug[1, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
                        img_aug[2, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
                        mask_aug[x[i]:x[i]+hh, y[i]:y[i]+ww] = 255

        img = img_aug.swapaxes(0,2)
        mask = mask_aug.swapaxes(0,1)
        
        return img, mask
        
    
class Target_Augmentation2(object):
   
     # Based on Random Erasing
    
    def __init__(self,  sl = 0.02, sh = 0.4, r1 = 0.3, numbers = 1): 
          
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.target_numbers = numbers
          
    def __call__(self, img, mask):

        area = 8 * 8 
        
        w = []
        h = []
        
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1/self.r1)
        
        for i in range(self.target_numbers):
            
            h.append(int(round(math.sqrt(target_area * aspect_ratio))))
            w.append(int(round(math.sqrt(target_area / aspect_ratio)))) 
            
        img_aug = np.array(img)    
        mask_aug = np.array(mask)
        
        img_aug = img_aug.swapaxes(0,2)
        mask_aug = mask_aug.swapaxes(0,1)
        
        x = []
        y = []
        
        if img_aug.shape[0]== 3:
            
            for i in range(self.target_numbers):

                if w[i] < img_aug.shape[1] and h[i] < img_aug.shape[2]:
                    
                    x.append(random.randint(0, img_aug.shape[1] - h[i]))
                    y.append(random.randint(0, img_aug.shape[2] - w[i]))
                    
                if img_aug[0,x[i],y[i]] > 0:
                    
                    m1, m2 = np.mgrid[:h[i], :w[i]]
                    circle = (m2 - h[i] // 2) ** 2 + (m1 - w[i] // 2) ** 2

                    target = -target + np.mean((img_aug)[0])
                    target[target < np.min((img_aug)[0])] = np.min((img_aug)[0])
                    theta = int(np.min((img_aug)[0]) / np.max((img_aug)[0]) * np.mean((img_aug)[0]))
                    target[target >= theta] = np.max((img_aug)[0])

                    hh = target.shape[0]
                    ww = target.shape[1]

                    img_aug[0, x[i]:x[i] + hh, y[i]:y[i] + ww] = target
                    img_aug[1, x[i]:x[i] + hh, y[i]:y[i] + ww] = target
                    img_aug[2, x[i]:x[i] + hh, y[i]:y[i] + ww] = target
                    mask_aug[x[i]:x[i] + hh, y[i]:y[i] + ww] = 255

        img = img_aug.swapaxes(0,2)
        mask = mask_aug.swapaxes(0,1)
        
        return img, mask
    
    
