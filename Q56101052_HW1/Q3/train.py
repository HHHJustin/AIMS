import torch
import sys
from construct_dataset import *
from torch.utils.data import DataLoader
from unet_model import UNet
from torch import  optim
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler

train_root = "..\\EKG_seg\\train"
train_dataset = heart_beat_dataset(train_root)
train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)
model = UNet(3,3)
lr = 0.01
epoch = 100
optimizer = optim.Adam(model.parameters(),lr = lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [15,30,45,60,75,90])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device,dtype=torch.float32)
criterion = nn.BCELoss()

for i in range(epoch):
    print("Epoch",i)
    for batch,(x,y) in enumerate(train_dataloader):
        print("Batch",batch)
        model.train()
        x = x.float()
        label = y['mask'].to(device,dtype = torch.float32)
        save = label[0,1,:,:]
        save = np.array(save.cpu())
        output = model(x.to(device,dtype = torch.float32))
        # background_loss = criterion(output[:,0,:,:],label[:,0,:,:])
        green_loss = criterion(output[:,1,:,:],label[:,1,:,:])
        red_loss = criterion(output[:,2,:,:],label[:,2,:,:])
        loss =  green_loss + red_loss
        print("Loss",loss,"Learning rate",lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    lr = scheduler.get_lr()
torch.save(model, 'u_net_2.pth')