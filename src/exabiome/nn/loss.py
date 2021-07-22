import torch
import torch.nn as nn


class DistMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Computes the phylogenetic distance loss

        Parameters
        ----------
        output
            the output of a network

        target
            the square root of the patristic distances
        """
        x2 = output.pow(2).sum(axis=1)
        xy = 2*output.mm(output.T)
        dist = (((x2 - xy).T + x2))
        n = output.shape[0]
        loss = (dist - target).pow(2).sum()/(n*(n-1))
        #loss = ((dist - target)/target.exp()).pow(2).sum()/(n*(n-1))
        return loss
