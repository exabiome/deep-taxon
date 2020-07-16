import torch
import torch.nn as nn


class DistMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        x2 = output.pow(2).sum(axis=1)
        xy = 2*output.mm(output.T)
        dist = (((x2 - xy).T + x2))
        n = output.shape[0]
        loss = (dist - target).pow(2).sum()/(n*(n-1))
        return loss
