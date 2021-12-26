'''
loss function
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Criterion():
    def __init__(self, lossFunction):
        self.lossFunction = lossFunction
    
    def getLossFunction(self, weight=None, size_average=True):
        if(self.lossFunction == 'CategoricalCrossEntropyLoss'):
            return CategoricalCrossEntropyLoss(weight=weight, size_average=size_average)
    
    def getActivateFunction(self, weight=None, size_average=True):
        if(self.lossFunction == 'CategoricalCrossEntropyLoss'):
            return CategoricalCrossEntropyLoss(weight=weight, size_average=size_average).getActivateF()


class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        #self.loss = nn.NLLLoss(weight=weight, size_average=size_average)
        self.loss = nn.L1Loss(reduction='mean')
        #self.loss = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    
    @staticmethod
    def getActivateF():
        return F.softmax

    def forward(self, outputs, targets):
        #outputs = F.log_softmax(outputs, dim=1)
        #outputs = F.softmax(outputs, dim=1)
        return self.loss(outputs, targets)