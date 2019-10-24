import h5py
from scipy.spatial.distance import squareform

from sklearn.manifold import MDS
from sklearn.preprocessing import normalize

import sys
import argparse
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('dist_h5', type=str, help='the HDF5 file with distance data (output by tree2dmat)')
parser.add_argument('out_h5', type=str, help='the HDF5 file to save the embedding to')
parser.add_argument('-p', '--n_components', type=int, default=2, help='the number of components to use')
parser.add_argument('-m', '--metric', action='store_true', default=False, help='perform metric MDS')
parser.add_argument('-n', '--normalize', action='store_true', default=False,
                    help='normalize samples after computing distances')
args = parser.parse_args()
logger = logging.getLogger()


logger.info('reading %s' % args.dist_h5)
with h5py.File(args.dist_h5, 'r') as f:
    dist = f['distances'][:]
    names = f['leaf_names'][:].astype('U')

logger.info('computing squareform')
dist = squareform(dist)
logger.info('computing %d components with %s MDS' % (args.n_components, "metric" if args.metric else "non-metric"))
mds = MDS(dissimilarity='precomputed', metric=args.metric, n_components=args.n_components)
emb = mds.fit_transform(dist)

if args.normalize:
    logger.info("normalizing samples")
    emb = normalize(emb, norm='l2', axis=1)

logger.info('saving embedding to %s' % args.out_h5)
with h5py.File(args.out_h5, 'w') as f:
    dset = f.create_dataset('embedding', data=emb)
    dset = f.create_dataset('leaf_names', shape=names.shape, dtype=h5py.special_dtype(vlen=str))
    dset[:] = names
