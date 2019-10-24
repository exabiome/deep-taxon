import os
import io
import sys
import numpy as np
import h5py
import glob
from datetime import datetime

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
samples = np.sort(samples)

logger.info('retrieving distances')
idx = np.zeros(args.n_taxa*(args.n_taxa-1)//2, dtype=int)
k = 0
n = taxa_names.shape[0]
for s_i, i in enumerate(samples):
    shift = i*n - ((i+1)*(i+2)//2)
    for s_j, j in enumerate(samples[s_i+1:]):
        idx[k] = shift + j
        k += 1
dist = dist[idx]

logger.info('retrieving names')
taxa_names = taxa_names[samples]

logger.info('retrieving embeddings')
emb = emb[samples]

logger.info('shearing tree')
tree_names = [s.replace('_', ' ') for s in taxa_names]
tree = tree.shear(tree_names)

logger.info('getting NCBI paths')
new_names = list()
fasta_dirs = list()
for _ in taxa_names:
    t = _[3:]
    new_names.append(t)
    dirs = ["genomes", "all", t[0:3]]
    dirs.extend(t[x:x+3] for x in range(4, 13, 3))
    dirs.append("%s*" % t)
    wc = os.path.join(*dirs)
    directory = glob.glob(wc)[0]
    fasta_dirs.append(directory)

if not args.no_gtdb_trim:
    logger.info('removing GTDB prefix')
    taxa_names = np.array(new_names)
    name_map = dict(zip(tree_names, new_names))
    for tip in tree.tips():
        before = tip.name
        tip.name = name_map[tip.name].replace('_', ' ')

logger.info('converting tree to Newick string')
bytes_io = io.BytesIO()
tree.write(bytes_io, format='newick')
tree_str = bytes_io.getvalue()

logger.info('writing data to %s' % args.out_h5)
with h5py.File(args.out_h5, 'w') as f:
    f.attrs['seed'] = args.seed
    import numpy as np
    f.create_dataset(DISTANCES, data=dist)
    f.create_dataset(EMBEDDING, data=emb)
    strtype = h5py.special_dtype(vlen=str)
    dset = f.create_dataset(LEAF_NAMES, shape=taxa_names.shape, dtype=strtype)
    dset[:] = taxa_names
    dset = f.create_dataset(FASTA_PATHS, shape=taxa_names.shape, dtype=strtype)
    dset[:] = fasta_dirs
    dset = f.create_dataset(TREE, shape=None, data=tree_str, dtype=h5py.special_dtype(vlen=bytes))
