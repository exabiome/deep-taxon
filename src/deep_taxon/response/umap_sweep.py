from deep_taxon.response.tree import upgma_tree
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import numpy as np
from .tree import get_phylo_stats



def plot_species_embedding(embedding, metadata, **kwargs):
    ax = sns.scatterplot(embedding[:,0], embedding[:,1], **kwargs)
    leg = ax.legend(loc='lower right', bbox_to_anchor=(1.40, 0.39))
    return ax, leg


def emb_tree(embedder, dist, leaf_names, target_tree, taxa_metadata, metric='euclidean', metric_kwargs=dict(), **fit_kwargs):
    emb = embedder.fit_transform(dist, **fit_kwargs)
    _dist = pdist(emb, metric=metric, **metric_kwargs)
    tree = upgma_tree(DistanceMatrix(squareform(_dist), leaf_names))
    ret = dict()
    ret['rfd'] = target_tree.compare_rfd(tree)
    ret['subsets'] = target_tree.compare_subsets(tree)
    ret['tip_distances'] = target_tree.compare_tip_distances(tree)
    ret['pearson'] = spearmanr(_dist, squareform(dist))[0]
    ret['spearman'] = pearsonr(_dist, squareform(dist))[0]
    ret.update(get_phylo_stats(tree, taxa_metadata))
    return ret


if __name__ == '__main__':

    from collections import OrderedDict
    from .tree import read_tree, get_dmat
    from .embedding import read_distances, read_embedding
    from .gtdb import read_taxonomy
    from ..utils import int_list, float_list, parse_seed
    from sklearn.model_selection import ParameterGrid
    from sklearn.preprocessing import LabelEncoder
    from umap import UMAP
    import pandas as pd
    import multiprocessing as mp
    import logging

    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()
    parser.add_argument('distances', type=str)
    parser.add_argument('target_tree', type=str)
    parser.add_argument('metadata', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-s', '--seed', type=parse_seed, default='')
    parser.add_argument('-p', '--n_components', type=int_list, default=[2])
    parser.add_argument('-n', '--n_neighbors', type=int_list, default=[50])
    parser.add_argument('-d', '--min_dist', type=float_list, default=[1.0])
    parser.add_argument('-w', '--target_weight', type=float_list, default=[0.01])
    parser.add_argument('-U', '--unsupervised', action='store_true', default=False)
    parser.add_argument('-r', '--tax_rank', choices=['phylum', 'class', 'order', 'family', 'genus'], default='phylum')
    parser.add_argument('-i', '--n_iters', default=1, type=int)
    parser.add_argument('-P', '--n_procs', default=1, type=int)

    args = parser.parse_args()

    logger = None
#    if args.n_procs > 1:
#        logger = mp.get_logger()
#        #hdlr = logging.StreamHandler(sys.stderr)
#        #hdlr.setLevel(logging.DEBUG)
#        #hdlr.setFormatter(logging.Formatter(fmt='%(processName)s - %(asctime)s - %(message)s'))
#        #logger.addHandler(hdlr)
#    else:
#        logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(message)s')
#        logger = logging.getLogger()
#
#    logger.info('HELLO WORLD')
#    print('FUCK THIS')


    dist, leaf_names = read_distances(args.distances, squareform=True)
    tree = read_tree(args.target_tree, leaf_names=leaf_names)
    mdf = read_taxonomy(args.metadata, leaf_names=leaf_names)

    params = dict(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    emb_kwargs = dict()
    if not args.unsupervised:
        params['target_metric'] = ['categorical']
        params['target_weight'] = args.target_weight
        emb_kwargs['y'] = LabelEncoder().fit_transform(mdf[args.tax_rank])

    grid = ParameterGrid(params)
    data = list()

    def func(fargs):
        p_id, p = fargs
        row = OrderedDict()
        row['param_id'] = p_id
        print(datetime.now().isoformat(), p_id, p)
        emb = UMAP(**p)
        row.update(emb.get_params())
        row.update(emb_tree(emb, dist, leaf_names, tree, mdf, **emb_kwargs))
        return row

    args_list = list()
    for p_id, p in enumerate(grid):
        for n in range(args.n_iters):
            p_copy = p.copy()
            p_copy['random_state'] = args.seed + p_id + n
            args_list.append((p_id, p_copy))

    map_func = map
    if args.n_procs > 1:
        pool = mp.Pool(args.n_procs)
        map_func = pool.imap

    def nstr(s):
        if s is None:
            return ''
        return str(s)

    it = map_func(func, args_list)
    out = open(args.output, 'w')
    count = 0
    try:
        line = next(it)
        print(',' + ','.join(line.keys()), file=out)
        print('0,' + ','.join(map(nstr, line.values())), file=out)
        for count, line in enumerate(it, 1):
            print(str(count) + ',' + ','.join(map(nstr, line.values())), file=out)
    finally:
        out.close()
