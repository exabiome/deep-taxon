import torch.nn as nn

class DistMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        x2 = (output**2).sum(axis=1)
        xy = 2*x.mm(output.T)
        dist = (((x2 - xy).T + x2)).sqrt()
        loss = self.mse(dist, target)
        return loss
