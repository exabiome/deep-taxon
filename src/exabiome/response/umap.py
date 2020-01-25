import h5py
from scipy.spatial.distance import squareform as _squareform

from umap import UMAP

from ..utils import parse_seed

from .embedding import read_distances, save_embedding


def run_umap(dist, logger=None, **kwargs):
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

    kwargs.setdefault('n_neighbors', 100)
    kwargs.setdefault('n_components', 2)

    if logger is not None:
        logger.info('computing {n_components} components with UMAP'.format(**kwargs))
        logger.info('using {n_neighbors} neighbors and {min_dist} min_dist'.format(**kwargs))


    umap = UMAP(metric='precomputed', **kwargs)
    emb = umap.fit_transform(dist)
    return emb


if __name__ == '__main__':

    import sys
    import argparse
    import logging
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('dist_h5', type=str, help='the HDF5 file with distance data (output by tree2dmat)')
    parser.add_argument('out_h5', type=str, help='the HDF5 file to save the embedding to')
    parser.add_argument('-p', '--n_components', type=int, default=2, help='the number of components to compute')
    parser.add_argument('-n', '--n_neighbors', type=int, default=100, help='the n_neighbors parameter of UMAP')
    parser.add_argument('-d', '--min_dist', type=float, default=0.1, help='the min_dist parameter of UMAP')
    parser.add_argument('-s', '--seed', type=parse_seed, default='', help='the seed to use for UMAP')
    parser.add_argument('--sqrt', action='store_true', default=False, help='square-root the distances before UMAP')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    logger.info('reading %s' % args.dist_h5)
    dist, names = read_distances(args.dist_h5)

    if args.sqrt:
        logger.info('taking square root of distances')
        dist = np.sqrt(dist)

    logger.info(f'using seed {args.seed}')

    emb = run_umap(dist, logger=logger,
                   random_state=args.seed,
                   n_components=args.n_components,
                   min_dist=args.min_dist,
                   n_neighbors=args.n_neighbors)

    logger.info('saving embedding to %s' % args.out_h5)
    save_embedding(args.out_h5, emb, names,
                   random_state=args.seed,
                   n_components=args.n_components,
                   min_dist=args.min_dist,
                   n_neighbors=args.n_neighbors)
