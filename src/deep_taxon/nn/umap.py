from warnings import warn

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import squareform
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from umap.umap_ import fuzzy_simplicial_set

from .samplers import DSSampler, WORSampler
from ..utils import log


def get_neighbor_graph(dmat, n_neighbors=15, random_state=None, labels=None):
    if len(dmat.shape) == 1:
        dmat = squareform(dmat)
    n_taxa = dmat.shape[0]

    if labels is not None:
        dmat = dmat[labels][:, labels]

    if n_neighbors > dmat.shape[0]:
        msg=f"Found fewer labels ({dmat.shape[0]}) than n_neighbors ({n_neighbors}). Reducing n_neighbors to {dmat.shape[0] - 1}"
        warn(msg)
        n_neighbors = dmat.shape[0] - 1

    graph, sigmas, rhos = fuzzy_simplicial_set(dmat,
                                               n_neighbors,
                                               random_state,
                                               metric='precomputed')
    if labels is not None:
        coo = graph.tocoo()
        graph = coo_matrix((coo.data, (labels[coo.row], labels[coo.col])),
                           shape=(n_taxa, n_taxa)).tocsr()

    return graph


    # genome_sizes = np.bincount(genome, weights=lengths)
def partition_neighbor_graph(graph, genome_sizes, size, rank):

    n_cc, components = connected_components(graph)

    genomes = list()
    orders = list()
    ordered_lengths = list()
    for i in range(n_cc):
        mask = components == i
        start_node = np.where(mask)[0][0]
        order, predecessor = breadth_first_order(umap_graph, start_node)
        orders.append(order)
        ordered_lengths.append(genome_sizes[order])

    ordered_lengths = np.concatenate(ordered_lengths)
    orders = np.concatenate(orders)

    # this could be more efficient, but idgaf
    mean_size = genome_sizes.sum() / size
    queries = np.arange(1, size + 1) * mean_size

    genome_sizes_cum = np.cumsum(ordered_lengths)
    ins_idx = np.searchsorted(genome_sizes_cum, queries)

    e = ins_idx[rank]
    s = 0 if rank == 0 else ins_idx[rank - 1]

    return orders[s:e]

def get_neighbor_graph_np(dmat, n_neighbors=15, labels=None):
    nn = np.argpartition(dmat, n_neighbors+1)[:, :n_neighbors+1]
    nn_dist = np.take_along_axis(dmat, nn, axis=1)
    nn_dist = np.argsort(nn_dist, axis=1)
    nn = np.take_along_axis(nn, nn_dist, axis=1)[:, 1:]

    rows = np.arange(dmat.shape[0])

    rho = dmat[rows, nn[:, 0]]
    sigma = dmat[rows, nn[:, -1]]

    row = np.repeat(rows, n_neighbors)
    col = nn.reshape(-1)

    graph = dok_matrix(dmat.shape)
    for i, j in zip(row, col):
        pij = np.exp((rho[i] - dmat[i, j]) / sigma[i])
        pji = np.exp((rho[j] - dmat[j, i]) / sigma[j])
        graph[i, j] = (pij + pji) - pij*pji

    graph = graph.tocsr()
    return graph


class UMAPLoss(nn.Module):

    def __init__(self, min_dist=0.01):
        super().__init__()
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
        p = torch.clamp(p, min=1e-4, max=1.0-1e-4)
        q = torch.clamp(q, min=1e-4, max=1.0-1e-4)
        ce = (p * (torch.torch.log(p/q)) + (1 - p) * (torch.log((1 - p)/(1 - q))))
        return ce.mean()

    def compute_distances(self, output1, output2):
        raise NotImplemented

    def forward(self, output, target_prob):
        """
        Computes the phylogenetic distance loss

        Parameters
        ----------
        output
            the output of a network

        target
            the square root of the patristic distances
        """
        output_from, output_to = output[::2], output[1::2]
        #cls_to, cls_from = target_cls[::2], target_cls[1::2]

        # compute distances - this will return the squareform
        dist = self.compute_distances(output_from, output_to)
        prob = 1.0 / (1.0 + self.a * dist ** (2 * self.b))

        return self.umap_loss(target_prob, prob)


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


class ContinuousSampler:

    def __init__(self, sampler, n_samples):
        self.sampler = sampler
        self.n_samples = n_samples
        self._it = None


    def __iter__(self):
        self.count = 0
        self._it = iter(self.sampler)
        return self

    def __next__(self):
        if self.count < self.n_samples:
            try:
                ret = next(self._it)
            except StopIteration:
                self._it = iter(self.sampler)
                ret = next(self._it)
            self.count += 1
            return ret
        raise StopIteration


class NeighborGraphSampler(Sampler):
    """
    Sample from a graph

    seq_labels/n_chunks_per_seq are assumed to be sorted by seq_label
    """

    def __init__(self, graph, seq_labels, n_chunks_per_seq, n_batches=1e4, batch_size=512, wor=True, rng=None):
        self.graph = graph.tocoo()
        if wor:
            self.edge_sampler = WORSampler(len(self.graph.data))
        else:
            self.edge_sampler = DSSampler(len(self.graph.data))
        self.__len = n_batches * batch_size // 2
        self.edge_sampler = ContinuousSampler(self.edge_sampler, self.__len)

        n_chunks_per_seq = 2 * n_chunks_per_seq
        counts = np.bincount(seq_labels, n_chunks_per_seq, minlength=graph.shape[0]).astype(int)
        mask = counts > 0
        self.labels = np.where(mask)[0]
        self.n_chunks_per_label = counts[mask]
        self.remaining = self.n_chunks_per_label.copy()

        # an index to get local label ID from a global ID
        self.inv_tax_map = -1 * np.ones(graph.shape[0], dtype=int)
        self.inv_tax_map[self.labels] = np.arange(len(self.labels))

        # continuously permuted sequence indices
        self.perm_index = np.cumsum(self.n_chunks_per_label)
        self.permutation = np.arange(self.perm_index[-1])

        self.rng = np.random.default_rng(rng)

        self._edges = None
        self._edge_id = None

    def get_chunk(self, label):
        # get remaining number of samples
        label = self.inv_tax_map[label]
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
        sample = self.rng.integers(wor_length)
        perm[[sample, -1]] = perm[[-1, sample]]

        return perm[-1]

    def __iter__(self):
        log("making iter from NeighborGraphSampler")
        self._edges = iter(self.edge_sampler)
        self._edge_id = None
        return self

    def __next__(self):
        if self._edge_id is None:
            self._edge_id = next(self._edges)
            ret = self.graph.row[self._edge_id]
        else:
            ret = self.graph.col[self._edge_id]
            self._edge_id = None
        label = ret
        ret = self.get_chunk(ret)
        return ret

    def __len__(self):
        return self.__len


class UMAPCollater:

    def __init__(self, graph, padval, seq_dtype=torch.uint8):
        self.graph = graph
        self.padval = padval
        self.seq_dtype = seq_dtype

    def __call__(self, samples):
        if isinstance(samples, tuple):
            samples = [samples]
        X_ret, y_ret = list(), list()
        idx_from, idx_to = list(), list()
        is_from = True
        for i, X, y, seq_id in samples:
            X_ret.append(X)
            y_ret.append(y)
            if is_from:
                idx_from.append(y)
            else:
                idx_to.append(y)
            is_from = not is_from
        X_ret = torch.stack(X_ret, out=torch.zeros(len(X_ret), len(X_ret[0]), dtype=self.seq_dtype))
        y_ret = torch.from_numpy(self.graph[idx_from, idx_to].getA()[0])
        return (X_ret, y_ret)
