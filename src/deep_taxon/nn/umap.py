from scipy.spatial.distance import squareform
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
from umap.umap_ import fuzzy_simplicial_set

from .samplers import DSSampler, WORSampler


def get_neighbor_graph(dmat, n_neightbors=15, random_state=None):
    if len(dmat.shape) == 1:
        dmat = squareform(dmat)
    graph, sigmas, rhos = fuzzy_simplicial_set(dmat,
                                               n_neighbors,
                                               random_state,
                                               metric='precomputed')
    return graph

class UMAPLoss(nn.Module):

    def __init__(self, graph, min_dist=0.01):
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
        output_to, output_from = output[::2], output[1::2]
        cls_to, cls_from = target_cls[::2], target_cls[1::2]
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


class ContinuousSampler(Sampler):

    def __init__(self, sampler, n_samples):
        self.sampler = sampler
        self.n_samples = n_samples


    def __iter__(self):
        self.count = 0
        self.it = iter(self.sampler)
        return self

    def __next__(self):
        if count < self.n_samples:
            try:
                ret = next(self.it)
            except StopIteration:
                self.it = iter(self.sampler)
                ret = next(self.it)
            return ret
        raise StopIteration


class NeighborGraphSampler(Sampler):
    """
    Sample from a graph

    seq_labels/n_chunks_per_seq are assumed to be sorted by seq_label
    """

    def __init__(self, graph, seq_labels, n_chunks_per_seq, n_batches=1e4, batch_size=512, wor=True):
        self.graph = graph.tocoo()
        if wor:
            self.edge_sampler = WORSampler(len(self.graph.data))
        else:
            self.edge_sampler = DSSampler(len(self.graph.data))
        self.edge_sampler = ContinuousSampler(self.edge_sampler, n_batches)

        counts = np.bincount(seq_labels, n_chunks_per_seq)
        mask = counts > 0
        self.labels = np.where(mask)[0]
        self.n_chunks_per_label = counts[mask]
        self.remaining = self.n_chunks_per_label.copy()

        # continuously permuted sequence indices
        self.perm_index = np.cumsum(self.n_chunks_per_label)
        self.permutation = np.arange(self.perm_index[-1])

        self.n_batches = n_batches
        self.batch_size = batch_size

    def get_chunk(self, label):
        # get remaining number of samples
        wor_length = self.remaining[label]
        if wor_length == 0:
            wor_length = self.n_chunks_per_label[label]
        self.remaining[label] = wor_length - 1

        # get start/stop of remaining samples
        start = 0 if label == 0 else self.perm_index[label - 1]
        end = start + wor_length

        # get remaining samples
        perm = self.permutation[start:end]

        # sample without replacement
        sample = rng.integers(wor_length)
        perm[[sample, -1]] = perm[[-1, sample]]

        return perm[-1]

    def __iter__(self):
        ret = list()
        for batch_i in range(self.n_batches):
            for i in range(len(self.batch_size) // 2):
                edge_id = next(self.edge_sampler)
                ret.append(self.get_chunk(self.graph.row[edge_id]))
                ret.append(self.get_chunk(self.graph.col[edge_id]))
            if len(ret) == self.batch_size:
                yield ret
                ret = list()
        raise StopIteration

    def __len__(self):
        return self.n_batches
