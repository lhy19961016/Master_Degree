import torch
import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import pandas as pd


from tqdm import tqdm

from utils import set_device

class XAI_DS(Dataset):
        def __init__(self, data_path, data_target, augmentation=None):
            super().__init__()
            self.data_path = data_path
            self.data_target = data_target
            self.augmentation = augmentation
        
        def __len__(self):
            return len(self.data_path)

        def __getitem__(self, idx):
            sample = torch.load(self.data_path[idx]).permute(2,0,1)
            if self.augmentation:
                input_tensor = self.augmentation(sample)
            else:
                 input_tensor = sample
            return input_tensor, torch.tensor(self.data_target[idx], dtype=torch.float)
        
import torch.nn as nn
import torch.nn.functional as F

class ResNetBasicBlock(nn.Module) :
    def __init__ (self, in_channels , out_channels, stride) :
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d (out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward (self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

class ResNetDownBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, stride) :
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn. Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn. Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
        nn.BatchNorm2d(out_channels))
    
    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
    
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2 , padding=0)
        
        self.layer1 = nn.Sequential(ResNetBasicBlock(64, 64, 1),
        ResNetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(ResNetDownBlock (64, 128, [2, 1]),
        ResNetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(ResNetDownBlock(128, 256, [2, 1]),
        ResNetBasicBlock(256, 256, 1) )
        self.layer4 = nn.Sequential(ResNetDownBlock (256, 512, [2, 1]),
        ResNetBasicBlock(512, 512, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 1) 
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape [0], -1)
        out = self.fc(out)
        return out

import copy

def train_model(model, loss, optimizer, scheduler, start_epochs, num_epochs, train_loader, val_loader):
    loss_best = 100000
    loss_hist = {'train':[], 'val':[]}
    acc_hist = {'train':[], 'val':[]}
 
    for epoch in range(start_epochs, num_epochs):
        print("Epoch {}/{}:\n".format(epoch, num_epochs - 1), end="")
        for phase in ['train', 'val']:
            if phase == 'train':  
                dataloader = train_loader
                scheduler.step()
                model.train()  
            else: 
                dataloader = val_loader 
                model.eval()  
            running_loss = 0. 
            running_acc = 0.
 
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(set_device()) 
                labels = labels.type(torch.LongTensor).to(set_device()) 
 
                optimizer.zero_grad()
 
                with torch.set_grad_enabled(phase == 'train'): 
                    preds = model(inputs) 
                    loss_value = loss(preds, labels)    
                    preds_class = preds.argmax(dim=1) 
                
                    if phase == 'train':
                        loss_value.backward() 
                        optimizer.step() 
                

                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean().data.cpu().numpy() 

            epoch_loss = running_loss / len(dataloader) 
            epoch_acc = running_acc / len(dataloader) 
            
            if phase == 'val':
                if epoch_loss < loss_best:
                    loss_best = epoch_loss
                    weight_best = copy.deepcopy(model.state_dict())

                    torch.save(weight_best, './Model/CheckPoint/best_weights_ddd2.pth')
                    model.load_state_dict(weight_best)
                    print("Best weights has been saved")
                    
            print("{} Loss: {:.4f} Acc: {:.4f} ".format(phase, epoch_loss, epoch_acc), end="")
            
            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)


    return model, loss_hist, acc_hist

def RESUME(model, optimizer):
    path_checkpoint = "./Model/CheckPoint/ckpt_best_30.pth" 
    checkpoint = torch.load(path_checkpoint) 

    model.load_state_dict(checkpoint['net']) 

    optimizer.load_state_dict(checkpoint['optimizer']) 
    start_epoch = checkpoint['epoch'] 
    # scheduler.load_state_dict(checkpoint['lr_schedule'])
    return model, optimizer, start_epoch

def main():
    resume = False
    train_transforms = transforms.Compose([
    # transforms.RandomPerspective(distortion_scale=0.09, p=0.75, fill=0), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5),  
    transforms.RandomRotation(45),
    # transforms.Normalize((-3.2940e-05 ,), (0.0098)), 
    # transforms.Normalize([-2.0809e-05, -4.8060e-05, -3.4650e-05], [0.0061, 0.0105, 0.0059]) // 4—2
    transforms.Normalize([-3.8328e-05, -9.1511e-05, -6.0215e-05], [0.0114, 0.0207, 0.0108])
    ])

    val_transforms = transforms.Compose([transforms.Normalize([-3.8328e-05, -9.1511e-05, -6.0215e-05], [0.0114, 0.0207, 0.0108])])
    Binary_classifier_XAI = Classifier()
    model = Binary_classifier_XAI
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to('cuda')
    # weights = torch.load("./Model/CheckPoint/ckpt_best_30.pth")['net']
    # model.load_state_dict(weights)

    df = pd.read_csv('./ddd2.csv')

    train_df, val_temp = train_test_split(df, test_size=0.4, random_state=45)
    val_df, test_df = train_test_split(val_temp, test_size=0.5, random_state=45)
    
    print(val_df[0:10])
    train_paths = [i for i in train_df["image"].values]
    train_target = train_df["label"].values

    valid_paths = [i for i in val_df["image"].values]
    valid_target = val_df["label"].values

    ds_train = XAI_DS(train_paths, train_target, train_transforms)
    ds_val = XAI_DS(valid_paths, valid_target, val_transforms)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0, last_epoch=-1, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1, verbose=True)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(ds_train , batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)

    start_epochs = -1

    if resume:
        model, optimizer, start_epochs = RESUME(model, optimizer)
    # optimizer.param_groups[0]['lr'] = 4.3227e-05

    model, loss, acc = train_model(model, loss, optimizer, scheduler, start_epochs=start_epochs ,num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader)
    
    checkpoint = {
        "net": model.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": num_epochs,
        "loss_curve": loss,
        "acc_curve": acc,
        'lr_schedule': scheduler.state_dict()
    }
    torch.save(checkpoint, './Model/CheckPoint/ckpt_best_%s_ddd2.pth' %(str(num_epochs)))


if __name__ == '__main__':
    main()
