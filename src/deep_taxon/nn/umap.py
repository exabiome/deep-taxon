from scipy.spatial.distance import squareform
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
from umap.umap_ import fuzzy_simplicial_set


class UMAPLoss(nn.Module):

    def __init__(self, dmat, n_neightbors=15, random_state=None, min_dist=0.01):
        graph, sigmas, rhos = fuzzy_simplicial_set(squareform(dmat),
                                                   n_neighbors,
                                                   random_state,
                                                   metric='precomputed')
        self.graph = graph
        self.a, self.b = self.find_ab_params(1.0, min_dist)

    @staticmethod
    def find_ab_params(spread, min_dist):
        """Fit a, b params for the differentiable curve used in lower
        dimensional fuzzy simplicial complex construction. We want the
        smooth curve (from a pre-defined family with simple gradient) that
        best matches an offset exponential decay.
        """

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, covar = curve_fit(curve, xv, yv)
        return params[0], params[1]

    @staticmethod
    def umap_loss(p, q):
        ce = (p * (torch.log(p/q)) + (1 - p) * (torch.log((1 - p)/(1 - q))))
        return ce.sum() / n

    def compute_distances(self, output1, output2):
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
        output_to, output_from = torch.chunk(output, 2)
        cls_to, cls_from = torch.chunk(target_cls, 2)
        target_prob = self.graph[cls_to, cls_from]

        target = self.graph[target_cls][:, target_cls]

        # compute distances - this will return the squareform
        dist = self.compute_distances(output)
        prob = 1.0 / (1.0 + self.a * dist ** (2 * self.b))

        return self.umap_loss(target, prob)


class HyperbolicUMAPLoss(UMAPLoss):

    def compute_distances(self, output1, output2):
    	s1 = torch.sqrt(1 + torch.sum(output1 ** 2, dim=1))
    	s2 = torch.sqrt(1 + torch.sum(output2 ** 2, dim=1))
    	B = s1 * s2
    	B -= torch.sum(output1 * output2 , dim=1)
    	B = torch.clamp(B, min=1.0 + 1e-8)
    	return torch.acosh(B)


class EuclideanUMAPLoss(UMAPLoss):

    def compute_distances(self, output1, output2):
        return torch.pow(output1 - output2, 2).sum(axis=1)
