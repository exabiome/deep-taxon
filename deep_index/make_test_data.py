import os
import io
import sys
import numpy as np
import h5py
import glob
from datetime import datetime
from scipy.spatial.distance import squareform

from sklearn.manifold import MDS

from skbio import TreeNode


import argparse
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(message)s')

DISTANCES = 'distances'
LEAF_NAMES = 'leaf_names'
EMBEDDING = 'embedding'
TREE = 'tree'
FASTA_PATHS = 'fasta_paths'

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('dist_h5', type=str, help='the HDF5 file with distance data (output by tree2dmat)')
parser.add_argument('emb_h5', type=str, help='the HDF5 file with embedding')
parser.add_argument('tree', type=str, help='the tree file to pull taxa from')
parser.add_argument('out_h5', type=str, help='the HDF5 file to save the embedding to')
parser.add_argument('-t', '--n_taxa', type=int, default=5, help='the number of taxa to sample')
parser.add_argument('-s', '--seed', type=int, default=None, help='the seed to use')
parser.add_argument('--no_gtdb_trim', action='store_true', default=False, help='do not trim GTDB prefix from names')

args = parser.parse_args()

logger = logging.getLogger()

if args.seed is None:
    args.seed = int(datetime.now().timestamp())

logger.info('using seed %d' % args.seed)
random_state = np.random.RandomState(args.seed)

logger.info('reading distances from %s' % args.dist_h5)
with h5py.File(args.dist_h5, 'r') as f:
    dist = f[DISTANCES][:]
    taxa_names = f[LEAF_NAMES][:].astype('U')

logger.info('reading embeddings from %s' % args.emb_h5)
with h5py.File(args.emb_h5, 'r') as f:
    emb = f[EMBEDDING][:]

logger.info('reading tree from %s' % args.tree)
tree = TreeNode.read(args.tree, format='newick')

logger.info('sampling %d of %d taxa' % (args.n_taxa, taxa_names.shape[0]))
samples = random_state.permutation(taxa_names.shape[0])[:args.n_taxa]     # sample taxa

logger.info('retreiving distances')
dist = squareform(dist)       # convert to squareform
dist = squareform(dist[samples,:][:,samples])      # convert back to condensed form
logger.info('retreiving names')
taxa_names = taxa_names[samples]
logger.info('retreiving embeddings')
emb = emb[samples]

for i, tip in enumerate(tree.tips()):
    if i == 5:
        break

logger.info('shearing tree')
tree_names = [s.replace('_', ' ') for s in taxa_names]
tree = tree.shear(tree_names)

logger.info('getting NCBI paths')
new_names = list()
for _ in taxa_names:
    t = _[3:]
    new_names.append(t)
    dirs = ["genomes", "all", t[0:3]]
    dirs.extend(t[x:x+3] for x in range(4, 13, 3))
    dirs.append("%s*" % t)
    wc = os.path.join(*dirs)
    directory = glob.glob(wc)[0]
    print(directory)



if not args.no_gtdb_trim:
    logger.info('removing GTDB prefix')
    taxa_names = np.array(new_names)
    name_map = dict(zip(tree_names, new_names))
    for tip in tree.tips():
        tip.name = name_map[tip.name].replace('_', ' ')

logger.info('converting tree to Newick string')
bytes_io = io.BytesIO()
tree.write(bytes_io, format='newick')
tree_str = bytes_io.read()

logger.info('writing data to %s' % args.out_h5)
with h5py.File(args.out_h5, 'w') as f:
    f.attrs['seed'] = args.seed
    f.create_dataset(DISTANCES, data=dist)
    f.create_dataset(EMBEDDING, data=emb)
    strtype = h5py.special_dtype(vlen=str)
    dset = f.create_dataset(LEAF_NAMES, shape=taxa_names.shape, dtype=strtype)
    dset[:] = taxa_names
    dset = f.create_dataset(FASTA_PATHS, data=dirs, shape=taxa_names.shape, dtype=strtype)
    dset = f.create_dataset(TREE, shape=None, data=tree_str, dtype=h5py.special_dtype(vlen=bytes))
