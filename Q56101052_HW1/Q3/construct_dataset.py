import PIL
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
from os.path import join
from os import listdir
import numpy as np

class heart_beat_dataset(Dataset):
    def __init__(self,root):
        self.root = root 
        self.img_transform = T.Compose([T.ToTensor(),
                                        T.Resize((512,512)),                                    
                                        T.Normalize([0.5,0.5,0.5], 
                                                    [0.5,0.5,0.5])])
        self.mask_transform = T.Compose([T.ToTensor()])
        self.file_list = listdir(self.root)
        
    def __getitem__(self, idx):
        img = cv2.imread(join(self.root,self.file_list[idx],'img.png'))
        data_name = self.file_list[idx].split('.')[0]
        w,h = img.shape[0],img.shape[1]
        
        img = self.img_transform(img)
    
        mask = cv2.imread(join(self.root,self.file_list[idx],'label.png'))
        mask = cv2.resize(mask,(512,512))
        mask_background = np.where(mask[:,:,0]>50,1,0) 
        mask_green = np.where(mask[:,:,1]>50,1,0) 
        mask_red = np.where(mask[:,:,2]>50,1,0) 
        temp = np.zeros((mask.shape[0],mask.shape[1],3))
        temp[:,:,0] = mask_background
        temp[:,:,1] = mask_green
        temp[:,:,2] = mask_red
        mask = temp
        mask = self.mask_transform(mask)
        label = {}
        label['name'] = data_name
        label['img_size'] = [w,h]
        label['mask'] = mask
        return img,label
        
    
    def __len__(self):
        return len(self.file_list)