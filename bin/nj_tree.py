import h5py
import sys
import argparse
import logging

from datetime import datetime
from scipy.spatial.distance import squareform, pdist
from skbio.tree import nj
from skbio.stats.distance import DistanceMatrix
from skbio import TreeNode

from sklearn.preprocessing import normalize


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

logger.info("reading data from %s" % args.emb_h5)
with h5py.File(args.emb_h5, 'r') as f:
    emb = f['embedding'][:]
    taxa = f['leaf_names'][:].astype('U').tolist()

target_tree = TreeNode.read(args.target_tree, format='newick')

if args.normalize:
    logger.info("normalizing samples")
    emb = normalize(emb, norm='l2', axis=1)

logger.info("computing %s distances" % args.metric)
dist = squareform(pdist(emb, metric=args.metric))

dmat = DistanceMatrix(dist, taxa)
logger.info("computing neighbor-joining tree")
tree = nj(dmat)
logger.info("done")

logger.info("comparing trees")
top_sim = target_tree.compare_subsets(tree)
blen_sim = target_tree.compare_tip_distances(tree)
logger.info(f"done. topology similarity: {top_sim} branch length similarity: {blen_sim}")


