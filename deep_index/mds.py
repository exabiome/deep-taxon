import h5py
from scipy.spatial.distance import squareform

from sklearn.manifold import MDS

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dist_h5', type=str, help='the HDF5 file with distance data (output by tree2dmat)')
parser.add_argument('out_h5', type=str, help='the HDF5 file to save the embedding to')
parser.add_argument('-p', '--n_components', type=int, default=2, help='the number of components to use')
parser.add_argument('-m', '--metric', action='store_true', default=False, help='perform metric MDS')
parser.add_argument('-n', '--normalize', action='store_true', default=False,
                    help='normalize samples before computing distances')
args = parser.parse_args()


with h5py.File(args.dist_h5, 'r') as f:
    dist = f['distances'][:]
    names = f['leaf_names'][:].astype('U')

dist = squareform(dist)
mds = MDS(dissimilarity='precomputed', metric=args.metric)
emb = mds.fit_transform(dist)

if args.normalize:
    print(datetime.now().isoformat(), "normalizing samples")
    emb = normalize(emb, norm='l2', axis=1)

with h5py.File(args.out_h5, 'w') as f:
    dset = f.create_dataset('embedding', data=emb)
    dset = f.create_dataset('leaf_names', shape=names.shape, dtype=h5py.special_dtype(vlen=str))
    dset[:] = names
