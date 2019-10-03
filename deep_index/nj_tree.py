import h5py
import argparse
from datetime import datetime
from scipy.spatial.distance import squareform, pdist
from skbio.tree import nj
from skbio.stats.distance import DistanceMatrix
from skbio import TreeNode

from sklearn.preprocessing import normalize


def biopy_tree(emb):
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
    dist = squareform(pdist(emb))
    tmp = list()
    for i in range(dist.shape[0]):
        tmp.append(dist[i, 0:i+1].tolist())
    dist = tmp
    dmat = DistanceMatrix(names=taxa, matrix=dist)

parser = argparse.ArgumentParser()
parser.add_argument('emb_h5', type=str, help='the HDF5 file with embedding')
parser.add_argument('target_tree', type=str, help='the tree file to compare to')
parser.add_argument('-n', '--normalize', action='store_true', default=False,
                    help='normalize samples before computing distances')
parser.add_argument('-m', '--metric', choices=['euclidean', 'mahalanobis', 'cosine'], default='euclidean',
                    help='the metric to use for computing distances from embeddings')
args = parser.parse_args()

with h5py.File(args.emb_h5, 'r') as f:
    emb = f['embedding'][:]
    taxa = f['leaf_names'][:].astype('U').tolist()

#with open(args.target_tree,'r') as f:
#    target_nwk = f.read()[:-1]
#
#target_tree = TreeNode.from_newick(target_nwk)
target_tree = TreeNode.read(args.target_tree, format='newick')

if args.normalize:
    print(datetime.now().isoformat(), "normalizing samples")
    emb = normalize(emb, norm='l2', axis=1)

print(datetime.now().isoformat(), "computing %s distances" % args.metric)
dist = squareform(pdist(emb, metric=args.metric))

dmat = DistanceMatrix(dist, taxa)
print(datetime.now().isoformat(), "computing neighbor-joining tree")
tree = nj(dmat)
print(datetime.now().isoformat(), "done")

print(datetime.now().isoformat(), "comparing trees")
top_sim = target_tree.compare_subsets(tree)
blen_sim = target_tree.compare_tip_distances(tree)
print(datetime.now().isoformat(), "done. topology similarity:", top_sim, " branch length similarity:", blen_sim)


