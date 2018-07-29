import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class FaceDataset(Dataset):

    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        
        self.d = {}
        with open(os.path.join(self.root_path, 'pnet/train_12.txt'), 'r') as f:
            for line in f:
                line = line.rstrip()
                impath, label, *reg = line.split()
                reg, label = np.array(list(map(float, reg))), int(label)
                if label == -1:
                    continue
                if label == 0:
                    reg = np.array([0., 0., 0., 0.])
                self.d[os.path.join(self.root_path, impath+'.jpg')] = (label, reg)
        self.imnames = list(self.d.keys())
    

    def __len__(self):
        return len(self.imnames)

    def __getitem__(self, idx):
        img_name = self.imnames[idx]
        
        label, reg = self.d[img_name]
        image = cv2.imread(img_name)
        
        if self.transform:
            image = self.transform(image)
            # put label transform here if needed

        return image, label, torch.from_numpy(reg)
    
    
    
class WiderDataset(Dataset):

    def __init__(self, anno_path, img_path, transform=None):
        self.anno_path = anno_path
        self.img_path = img_path
        self.transform = transform
        
        self.d = {}
        with open(anno_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                imname, *reg = line.split()
                reg = np.array(list(map(float, reg))).reshape(-1, 4)
                self.d[os.path.join(self.img_path, imname)] = reg
        self.imnames = list(self.d.keys())
    

    def __len__(self):
        return len(self.imnames)

    def __getitem__(self, idx):
        img_name = self.imnames[idx]
        
        
        reg = self.d[img_name]
        image = cv2.imread(img_name)
        
        if self.transform:
            image = self.transform(image)
            # put label transform here if needed

        return image, torch.from_numpy(reg)