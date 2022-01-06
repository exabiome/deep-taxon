import torch
import torch.nn as nn


class EuclideanMAELoss(nn.Module):

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
        loss = (dist - target).abs().tril(diagonal=-1).sum()/(n*(n-1))
        return loss


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


# class HyperbolicDistortionLoss(nn.Module):
#
#     def __init__(self, tol=1e-6, k=5):
#         super().__init__()
#         self.tol = tol
#         self.k = k
#         self._mean = torch.tensor((k+1)/2)
#
#         self._var = torch.tensor((k + 1) * (2 * k + 1) / 6) - self._mean**2
#
#     def forward(self, output, target):
#         """
#         Computes the phylogenetic distance loss
#
#         Parameters
#         ----------
#         output
#             the output of a network
#
#         target
#             the square root of the patristic distances
#         """
#
#         output = output.double()    # might not need this, since we just need it for kNN
#         # compute hyperbolic distance
#         s = torch.sqrt(1 + torch.sum(output ** 2, dim=1))
#         B = torch.outer(s, s)
#         B -= output.matmul(output.T)
#         B[(B - 1.0).abs() < self.tol] = 1.0
#         dist = torch.acosh(B)
#
#         # dev_o = dist.topk(self.k + 1, largest=False).indices[:, 1:] - self._mean
#         # dev_t = target.topk(self.k + 1, largest=False).indices[:, 1:] - self._mean
#         # pearson = ((dev_o * dev_t).sum()/self.k)/self._var
#         # loss = pearson
#
#         loss = ((dist.topk(output.shape[0], largest=False).indices[:, 1:] - target.topk(output.shape[0], largest=False).indices[:, 1:])**2).float().mean()
#         return loss


class HyperbolicMAELoss(nn.Module):

    def __init__(self, tol=1e-6):
        super().__init__()
        self.tol = tol

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
        # compute hyperbolic distance
        output = output.double()
        s = torch.sqrt(1 + torch.sum(output ** 2, dim=1))
        B = torch.outer(s, s)
        B -= output.matmul(output.T)
        B[(B - 1.0).abs() < self.tol] = 1.0
        dist = torch.acosh(B)

        # compute mean absolute error
        n = output.shape[0]
        loss = (dist - target).abs().tril(diagonal=-1).sum()/(n*(n-1))
        return loss
