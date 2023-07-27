import torch
import sys
from construct_dataset import *
from torch.utils.data import DataLoader
from unet_model import UNet
from torch import  optim
import torch.nn as nn
import numpy as np
import pandas as pd
from os.path import join
import cv2

test_root = "..\\EKG_seg\\test"
test_dataset = heart_beat_dataset(test_root)
test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False)
model_path = "u_net_2.pth"
model = torch.load(model_path)
model.cpu()
model.eval()

for batch,(x,y) in enumerate(test_dataloader):
    output = model(x)
    size = y['img_size']
    data_name = y['name'][0]
    green = output[0,1,:,:]
    green = np.where(green > 0.5,1,0)*255
    
    red = output[0,2,:,:]
    red = np.where(red > 0.5,1,0)*255
    img = np.zeros((3,512,512))
    img[1,:,:] = green
    img[2,:,:] = red
    img = np.array(img).astype(np.uint8)
    
    img = img.transpose((1,2,0))
    img = cv2.resize(img,(int(size[1]),int(size[0])))
    
    cv2.imwrite(str(data_name) + '.png',img)
    
    