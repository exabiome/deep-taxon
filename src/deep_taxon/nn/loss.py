import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


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


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output
