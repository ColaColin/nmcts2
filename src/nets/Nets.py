'''
Created on Apr 3, 2018

@author: cclausen
'''

import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        
        out = self.act(out)
        
        return out
    
class ResCNN(nn.Module):
    def __init__(self, inWidth, inHeight, inDepth, baseKernelSize, baseFeatures, features, blocks, moveSize, winSize):
        super(ResCNN, self).__init__()
        assert (inWidth % 2) == (inHeight % 2)
        
        self.baseConv = nn.Conv2d(inDepth, baseFeatures, baseKernelSize)
        self.baseBn = nn.BatchNorm2d(baseFeatures)
        self.act = nn.ReLU(inplace=True)
        
        if (baseFeatures != features and blocks > 0):
            self.matchConv = nn.Conv2d(baseFeatures, features, 1)
        else:
            self.matchConv = None
        
        blockList = []
        for _ in range(blocks):
            blockList.append(ResBlock(features))
        self.resBlocks = nn.Sequential(*blockList)

        hiddens = features * (inWidth - (baseKernelSize - 1)) * (inHeight - (baseKernelSize - 1))
        self.moveHead = nn.Linear(hiddens, moveSize)
        
        if winSize > 0:
            self.winHead = nn.Linear(hiddens, winSize)
        else:
            self.winHead = None
            
        self.lsoftmax = nn.LogSoftmax()
    
    def forward(self, x):
        x = self.act(self.baseBn(self.baseConv(x)))
        
        if (self.matchConv != None):
            x = self.matchConv(x)
        
        x = self.resBlocks(x)
        
        x = x.view(x.size(0), -1)

        moveP = self.lsoftmax(self.moveHead(x))
        
        if self.winHead != None:
            winP = self.lsoftmax(self.winHead(x))
            return moveP, winP
        else:
            return moveP
        
class MLP(nn.Module):
    def __init__(self, inSize, hiddens, moveSize, winSize):
        super(MLP, self).__init__()
        self.h = nn.Linear(inSize, hiddens)
        self.moveHead = nn.Linear(hiddens, moveSize)
        self.winHead = nn.Linear(hiddens, winSize)
        self.hact = nn.ReLU()
        self.lsoftmax = nn.LogSoftmax()
        
    def forward(self, x):
        x = self.hact(self.h(x))
        return self.lsoftmax(self.moveHead(x)), self.lsoftmax(self.winHead(x))