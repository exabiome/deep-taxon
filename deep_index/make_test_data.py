import os
import sys
import numpy as np
import h5py
from datetime import datetime
from scipy.spatial.distance import squareform

from sklearn.manifold import MDS

import argparse

DISTANCES = 'distances'
LEAF_NAMES = 'leaf_names'
EMBEDDING = 'embedding'

parser = argparse.ArgumentParser()
parser.add_argument('dist_h5', type=str, help='the HDF5 file with distance data (output by tree2dmat)')
parser.add_argument('emb_h5', type=str, help='the HDF5 file with embedding')
parser.add_argument('out_h5', type=str, help='the HDF5 file to save the embedding to')
parser.add_argument('-t', '--n_taxa', type=int, default=5, help='the number of taxa to sample')
parser.add_argument('-s', '--seed', type=int, default=None, help='the seed to use')
args = parser.parse_args()

if args.seed is None:
    args.seed = int(datetime.now().timestamp())

random_state = np.random.RandomState(args.seed)

with h5py.File(args.dist_h5, 'r') as f:
    dist = f[DISTANCES][:]
    taxa_names = f[LEAF_NAMES][:].astype('U')

with h5py.File(args.emb_h5, 'r') as f:
    emb = f[EMBEDDING][:]

dist = squareform(dist)

samples = random_state.permutation(taxa_names.shape[0])[:args.n_taxa]

dist = squareform(dist[samples,:][:,samples])
taxa_names = taxa_names[samples]
emb = emb[samples]

with h5py.File(args.out_h5, 'w') as f:
    f.attrs['seed'] = args.seed
    f.create_dataset(DISTANCES, data=dist)
    f.create_dataset(EMBEDDING, data=emb)
    dset = f.create_dataset(LEAF_NAMES, shape=taxa_names.shape, dtype=h5py.special_dtype(vlen=str))
    dset[:] = taxa_names

for _ in taxa_names:
    t = _[3:]
    dirs = ["genomes", "all", t[0:3]]
    dirs.extend(t[x:x+3] for x in range(4, 13, 3))
    dirs.append("%s*" % t)
    print(os.path.join(*dirs))
