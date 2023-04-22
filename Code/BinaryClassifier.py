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
