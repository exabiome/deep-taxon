import matplotlib.pyplot as plt
import seaborn as sns

from exabiome.response.tree import read_tree, get_phylo_stats
from exabiome.response.embedding import read_distances

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

from umap import UMAP
from datetime import datetime

from scipy.sparse import lil_matrix, csr_matrix

import numpy as np


def _tally_dist(left, right, distances, dmat):
    for i in left:
        for j in right:
            dmat[i, j] = distances[i] + distances[j]
            dmat[j, i] = dmat[i, j]
    ret = left + right
    return ret


def accum_dist(node, distances, dmat, names):
    if len(node.children) == 0:
        distances[node.id] = node.length
        ret = [node.id]
        names[node.id] = node.name
    else:
        names[node.id] = node.name or "internode-%d" % node.id
        left = accum_dist(node.children[0], distances, dmat, names)
        right = accum_dist(node.children[1], distances, dmat, names)
        ret = _tally_dist(left, right, distances, dmat)
        dmat[node.id, ret] = distances[ret]
        dmat[ret, node.id] = distances[ret]
    return ret

def _add_dist(node, children, distances, dmat):
    dmat[node.id, children] = distances[children] + node.length
    dmat[children, node.id] = dmat[node.id, children]

def get_full_dmat(_tree):
    n_nodes = _tree.count(True) * 2 -2
    dmat = np.zeros((n_nodes, n_nodes))
    distances = np.zeros(n_nodes)
    names = [""] * n_nodes
    left_node = _tree.children[0]
    right_node = _tree.children[1]
    mid_blen = left_node.length

    left = accum_dist(left_node, distances, dmat, names)
    right = accum_dist(right_node, distances, dmat, names)

    distances[left] = distances[left] + right_node.length
    _tally_dist(left, right, distances, dmat)
    distances[left] = distances[left] - right_node.length

    dmat[left_node.id, right_node.id] = left_node.length
    dmat[right_node.id, left_node.id] = left_node.length
    distances[left_node.id] = mid_blen
    distances[right_node.id] = mid_blen

    _add_dist(right_node, left, distances, dmat)
    _add_dist(left_node, right, distances, dmat)

    return dmat, np.array(names)


if __name__ == '__main__':


    from six import StringIO
    from skbio import TreeNode
    tree = TreeNode.read(StringIO("((a:1,b:2)c:6,(d:4,e:5)f:6)root;")).unrooted_copy()
    tree.assign_ids()


    result = np.array([[ 0.,  3., 11., 12.,  1.,  7.],
                       [ 3.,  0., 12., 13.,  2.,  8.],
                       [11., 12.,  0.,  9., 10.,  4.],
                       [12., 13.,  9.,  0., 11.,  5.],
                       [ 1.,  2., 10., 11.,  0.,  6.],
                       [ 7.,  8.,  4.,  5.,  6.,  0.]])

    dmat, names = get_dmat(tree)
    np.testing.assert_equal(dmat, result)
    print(names)
    print(dmat)
