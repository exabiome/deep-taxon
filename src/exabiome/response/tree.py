import h5py
import sys
import argparse
import logging

from scipy.spatial.distance import squareform, pdist
from skbio.tree import nj
from skbio.stats.distance import DistanceMatrix
from skbio import TreeNode

from sklearn.preprocessing import normalize as _normalize

from .embedding import read_embedding


def get_dmat(embedding, leaf_names, metric='euclidean', logger=None):
    """
    Compute distances from embedding and return scikit-bio DistanceMatrix

    Args:
        embedding:          the embedding for each taxa
        leaf_names:         the leafe
    """
    if logger:
        logger.info("computing %s distances" % metric)
    dist = squareform(pdist(embedding, metric=metric))
    dmat = DistanceMatrix(dist, leaf_names)
    return dmat


def nj_tree(dmat):
    tree = nj(dmat)
    return tree


def read_tree(nwk_path, leaf_names=None):
    """
    Read a tree in Newick format

    Returns:
        TreeNode object for the root of the tree
    """
    tree = TreeNode.read(nwk_path, format='newick')
    for n in tree.tips():
        n.name = n.name.replace(' ', '_')
    if leaf_names is not None:
        tree = tree.shear(leaf_names)
    return tree


def compare_tree(tree, target_nwk_path):
    """
    Compare tree and tree from target_nwk_path

    Return:
        Tuple of topology similarity and branch-length similarity
    """
    target_tree = read_tree(target_nwk_path)
    top_sim = target_tree.compare_subsets(tree)
    blen_sim = target_tree.compare_tip_distances(tree)
    return top_sim, blen_sim


def get_tree(emb_h5, normalize=False, metric='euclidean', logger=None):
    if logger:
        logger.info("reading data from %s" % args.emb_h5)
    emb, leaf_names = read_embedding(emb_h5)
    if normalize:
        if logger:
            logger.info("normalizing samples")
        emb = _normalize(emb, norm='l2', axis=1)

    dmat = get_dmat(emb, leaf_names, metric=metric, logger=logger)
    if logger:
        logger.info("computing neighbor-joining tree")
    tree = nj_tree(dmat)
    if logger:
        logger.info("done")
    return tree



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_h5', type=str, help='the HDF5 file with embedding')
    parser.add_argument('target_tree', type=str, help='the tree file to compare to')
    parser.add_argument('-n', '--normalize', action='store_true', default=False,
                        help='normalize samples before computing distances')
    parser.add_argument('-m', '--metric', choices=['euclidean', 'mahalanobis', 'cosine'], default='euclidean',
                        help='the metric to use for computing distances from embeddings')
    args = parser.parse_args()
    logger = logging.getLogger()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')

    tree = get_tree(args.emb_h5, normalize=args.normalize, metric=args.metric, logger=logger)

    logger.info("comparing trees")
    top_sim, blen_sim = compare_tree(tree, args.target_tree)
    logger.info(f"done. topology similarity: {top_sim} branch length similarity: {blen_sim}")
