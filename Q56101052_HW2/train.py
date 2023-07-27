import torch
from torch.utils.data import Dataset,DataLoader
from os import listdir
from os.path import join
import pandas as pd
import torchvision.transforms as T
import PIL
from torch.nn.modules.loss import BCEWithLogitsLoss
import torchvision
from tensorboardX import SummaryWriter 

classes = {'Cardiomegaly': 0, 'No Finding': 1, 'Hernia': 2, 'Mass': 3, 'Infiltration': 4, 'Effusion': 5, 'Nodule': 6, 'Emphysema': 7, 'Atelectasis': 8, 'Pleural_Thickening': 9, 'Pneumothorax': 10, 'Fibrosis': 11, 'Consolidation': 12, 'Edema': 13, 'Pneumonia': 14}

class xray_dataset(Dataset):
    def __init__(self,root,training):
        self.root = root
        self.imgs = listdir(self.root)
        self.df = pd.read_csv('Data_Entry_2017.csv')
        self.training = training

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = join(self.root,img_name)
        img = PIL.Image.open(img_path).convert('RGB')
        transforms = T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        img = transforms(img)
        if self.training:
            flip = T.RandomHorizontalFlip(0.5)
            img = flip(img)
        check = (self.df['Image Index'] == img_name)
        if classes[list(self.df[check]["Finding Labels"])[0].split('|')[0]] == 1:
            label = 1
        else:
            label = 0
        label  = torch.tensor(label)
        
        return img,label

    def __len__(self):
        return len(self.imgs)

def main():
    model = torchvision.models.resnet101()
    model.fc = torch.nn.Linear(2048, 2)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model =model.to(device)
    train_path = 'training'
    val_path = 'testing'
    train_dataset = xray_dataset(train_path,training=True)
    val_dataset = xray_dataset(val_path,training=False)
    train_loader = DataLoader(train_dataset,batch_size=16,num_workers=4,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=16,num_workers=4,shuffle=False)
    lr = 0.001
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    epochs = 100
    lossFn = BCEWithLogitsLoss()
    trainSteps = len(train_loader.dataset) 
    valSteps = len(val_loader.dataset)
    writer = SummaryWriter()
    for e in range(0,epochs):
        model.train()
        if e == 30:
            lr = 0.0001
            opt = torch.optim.Adam(model.parameters(),lr = lr)
        if e == 70:
            lr = 0.00001
            opt = torch.optim.Adam(model.parameters(),lr = lr)
        totalTrainLoss = 0
        totalValLoss = 0 
        trainCorrect = 0
        valCorrect = 0
        for batch,(x,y) in enumerate(train_loader):
            sigmoid = torch.nn.Sigmoid()
            (x,y) = (x.to(device),y.to(device))
            pred =  model(x.float())
            label = torch.nn.functional.one_hot(y,2)
            loss = lossFn(pred,label.float())
            # loss = torchvision.ops.sigmoid_focal_loss(pred,label.float(),reduction="sum")
            opt.zero_grad()
            loss.backward()
            opt.step()
            totalTrainLoss += loss.item() 
            _, pred_out = torch.max(pred,1)
            num_correct = (pred_out == y).sum()
            trainCorrect += num_correct.item()
            if batch%50 == 0:
                print("Batch:",batch,"Loss",loss.item()/32)
            
        with torch.no_grad():
            model.eval()
            for (x,y) in val_loader:
                (x,y) = (x.to(device) , y.to(device))
                pred =  model(x.float())
                label = torch.nn.functional.one_hot(y,2)
                # loss = torchvision.ops.sigmoid_focal_loss(pred,label.float(),reduction="sum")
                loss = lossFn(pred,label.float())
                totalValLoss += loss.item()
                valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
                
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        trainCorrect = (trainCorrect / trainSteps) * 100
        valCorrect = (valCorrect / valSteps) * 100
        writer.add_scalar('Accuracy', valCorrect,e)
        writer.add_scalar('Loss',avgValLoss,e)
        
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}%".format(avgTrainLoss, trainCorrect ))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}%\n".format(avgValLoss, valCorrect))
    
        if e%10 == 0:
            torch.save(model, 'Save_model_'+str(e)+'.pt')
    torch.save(model, 'Save_model.pt')
    writer.close()

if __name__ == '__main__':
    main()