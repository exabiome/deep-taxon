import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class CondensedDistanceLoss(nn.Module):

    def __init__(self, dmat, batch_size):
        super().__init__()
        self.dmat = dmat
        if not isinstance(self.dmat, torch.Tensor):
            raise ValueError(f"CondensedDistanceLoss requires a Tensor, got {type(self.dmat)}")
        self.dmat /= self.dmat.max()

        self.device = dmat.device
        self.m = torch.tensor(int((1 + math.sqrt(1 + 8*len(dmat))) / 2), device=self.device)
        self.indices = torch.triu_indices(batch_size, batch_size, offset=1, device=self.device)
        self.batch_size = batch_size
        self.zero = torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def compute_distances(self, output):
        raise NotImplemented

    def forward(self, output, target_cls):
        """
        Computes the phylogenetic distance loss

        Parameters
        ----------
        output
            the output of a network

        target
            the square root of the patristic distances
        """
        indices = self.indices
        if len(target_cls) != self.batch_size:
            self.indices = torch.triu_indices(len(target_cls), len(target_cls), offset=1, device=self.device)

        # get sub-distance matrix from condensed distance matrix
        i, j = torch.sort(target_cls[indices], dim=0)[0]
        target = self.m * i + j - torch.div(((i + 2) * (i + 1)), 2, rounding_mode='trunc')
        target = self.dmat[target]
        target[i == j] = self.zero

        # compute distances and subset it to get condensed form
        dist = self.compute_distances(output)
        dist = dist[indices[0], indices[1]]

        loss = (dist - target).abs().mean()
        return loss


def euclidean_loss(output):
    x2 = output.pow(2).sum(axis=1)
    xy = 2*output.mm(output.T)
    dist = (((x2 - xy).T + x2))
    return dist


def hyperbolic_loss(output, tol=1e-6):
    # compute hyperbolic distance
    output = output.double()
    s = torch.sqrt(1 + torch.sum(output ** 2, dim=1))
    B = torch.outer(s, s)
    B -= output.matmul(output.T)
    B[(B - 1.0).abs() < tol] = 1.0
    dist = torch.acosh(B)
    return dist


class CondensedEuclideanMAELoss(CondensedDistanceLoss):

    def compute_distances(self, output):
        return euclidean_loss(output)


class CondensedHyperbolicMAELoss(CondensedDistanceLoss):

    def __init__(self, dmat, batch_size, tol=1e-6):
        super().__init__(dmat, batch_size)
        self.tol = torch.tensor(tol, device=self.device)

    def compute_distances(self, output):
        return hyperbolic_loss(output, tol=self.tol)


class EuclideanMAELoss(nn.Module):

    def forward(self, output, target):
        dist = euclidean_loss(output)
        return (dist - target).abs().mean()


class HyperbolicMAELoss(nn.Module):

    def __init__(self, tol=1e-6):
        super().__init__()
        self.tol = tol

    def forward(self, output, target):
        dist = hyperbolic_loss(output, tol=self.tol)
        return (dist - target).abs().mean()


class ArcMarginProduct(nn.Module):
    """Adapted from https://github.com/ronghuaiyang/arcface-pytorch

        Implement of large margin arc distance: :
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
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output
