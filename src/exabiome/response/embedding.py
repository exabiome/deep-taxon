import h5py
import pandas as pd
from scipy.spatial.distance import squareform as _squareform

from sklearn.manifold import MDS
from sklearn.preprocessing import normalize



def read_distances(dist_h5, squareform=False):
    """
    Read distances produced by tree2dmat
    Returns:
        tuple of distances and leaf names
    """
    with h5py.File(dist_h5, 'r') as f:
        dist = f['distances'][:]
        names = f['leaf_names'][:].astype('U')
    if squareform:
        dist = _squareform(dist)
    return dist, names


def save_embedding(out_h5, embedding, names, **kwargs):
    """
    Save an embedding file

    Args:
        out_h5:            the output HDF5 file
        embedding:         the embedding data
        names:             the name or accession for each sample
        kwargs:            any metadata to add to the file
    """
    with h5py.File(out_h5, 'w') as f:
        dset = f.create_dataset('embedding', data=embedding)
        dset = f.create_dataset('leaf_names', shape=names.shape, dtype=h5py.special_dtype(vlen=str))
        dset[:] = names
        for k, v in kwargs.items():
            f.attrs[k] = v


def read_embedding(emb_h5, df=False):
    """
    Read embeddings

    Args:
        emb_h5:         path to HDF5 file
        df:             return as a DataFrame. default is False

    Returns:
        tuple of embedding and leaf names
    """
    with h5py.File(emb_h5, 'r') as f:
        emb = f['embedding'][:]
        taxa = f['leaf_names'][:].astype('U').tolist()

    if df:
        colnames = ["dim%02g" % i for i in range(emb.shape[1])]
        df = pd.DataFrame(data=emb, columns=colnames, index=taxa)
        return df
    else:
        return emb, taxa


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
