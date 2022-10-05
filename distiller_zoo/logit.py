from turtle import forward
import torch.nn as nn
# import torch.nn.functional as F

class Loss(nn.Module):
    """ Cross Entropy Loss """
    def __init__(self):
        super(Loss, self).__init__()
        self.logloss = nn.nn.BCEWithLogitsLoss()
    
    def forward(self, y_s, label):
        loss = self.logloss(y_s, label.y.squeeze(-1))
        return loss