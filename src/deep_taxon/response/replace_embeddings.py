import deep_taxon.sequence
from hdmf.common import get_hdf5io
from deep_taxon.response.embedding import read_embedding

import pandas as pd

if __name__ == '__main__':

    import argparse


    parser = argparse.ArgumentParser(description='substitute new embeddings in a DeepIndex input file')
    parser.add_argument('new_embeddings', type=str, help='the embeddings to add to the DeepIndex input file')
    parser.add_argument('deep_index_input', type=str, help='the DeepIndex file with embeddings to overwwrite')
    args = parser.parse_args()

    emb, leaf_names = read_embedding(args.new_embeddings)
    leaf_names = [_[3:] for _ in leaf_names]

    emb_df = pd.DataFrame(data={'embedding1': emb[:,0], 'embedding2': emb[:,1]}, index=leaf_names)

    hdmfio = get_hdf5io(args.deep_index_input, mode='a')
    difile = hdmfio.read()
    di_taxa = difile.taxa_table['taxon_id'][:]
    di_emb = difile.taxa_table['embedding'].data   # h5py.Dataset

    emb_df = emb_df.filter(items=di_taxa, axis=0)

    di_emb[:] = emb_df.values

    hdmfio.close()
