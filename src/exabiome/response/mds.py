import h5py
from scipy.spatial.distance import squareform as _squareform

from sklearn.manifold import MDS
from sklearn.preprocessing import normalize

from .embedding import read_distances, save_embedding


def mds(dist, n_components=2, metric=False, logger=None):
    """
    Run MDS on distances produced by tree2dmat

    Args:
        dist (str):             A distance matrix, square or condensed form
        n_components (int):     number of components to produce
        metric (bool):          Whether or not to run metric MDS. default is to run non-metric
        logger (Logger):        Logger to use. default is no logging

    Return:
        emb (np.array):         the MDS embedding
    """
    if len(dist.shape) == 1:
        if logger is not None:
            logger.info('computing squareform')
        dist = _squareform(dist)

    if logger is not None:
        logger.info('computing %d components with %s MDS' % (n_components, "metric" if metric else "non-metric"))

    mds = MDS(dissimilarity='precomputed', metric=metric, n_components=n_components)
    emb = mds.fit_transform(dist)
    return emb


if __name__ == '__main__':

    import sys
    import argparse
    import logging
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('dist_h5', type=str, help='the HDF5 file with distance data (output by tree2dmat)')
    parser.add_argument('out_h5', type=str, help='the HDF5 file to save the embedding to')
    parser.add_argument('-p', '--n_components', type=int, default=2, help='the number of components to use')
    parser.add_argument('-m', '--metric', action='store_true', default=False, help='perform metric MDS')
    parser.add_argument('-s', '--sqrt', action='store_true', default=False, help='square-root the distances before MDS')
    parser.add_argument('-n', '--normalize', action='store_true', default=False,
                        help='normalize samples after computing distances')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    logger.info('reading %s' % args.dist_h5)
    dist, names = read_distances(args.dist_h5)

    if args.sqrt:
        logger.info('taking square root of distances')
        dist = np.sqrt(dist)

    emb = mds(dist, n_components=args.n_components, metric=args.metric, logger=logger)

    if args.normalize:
        logger.info("normalizing samples")
        emb = normalize(emb, norm='l2', axis=1)

    logger.info('saving embedding to %s' % args.out_h5)
    save_embedding(args.out_h5, emb, names)
