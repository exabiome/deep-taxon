from collections import Counter
import pandas as pd
import numpy as np
import skbio.io
import h5py


rcmap = ['' for i in range(128)]
rcmap[ord('A')] = 'T'
rcmap[ord('T')] = 'A'
rcmap[ord('C')] = 'G'
rcmap[ord('G')] = 'C'
rcmap[ord('N')] = 'N'
def ckmer(s):
    rc = ''.join(rcmap[ord(c)] for c in s[::-1])
    if s < rc:
        return s
    else:
        return rc


def get_tetramers(path, **kwargs):
    tnf = list()
    seqnames = list()
    for seq_i, seqobj in enumerate(skbio.io.read(path, 'fasta', **kwargs)):
        seq = str(seqobj)
        seqnames.append(seqobj.metadata['id'])
        tnf.append(Counter(str(seq[i:i+4]) for i in range(len(seq)-3)))
    df = pd.DataFrame(data=tnf, index=seqnames).fillna(0.0)

    to_drop = list()
    for kmer in df.columns:
        if 'N' in kmer:
            to_drop.append(kmer)
            continue
        ck = ckmer(kmer)
        if ck != kmer:
            to_drop.append(kmer)
            df[ck] = df[ck] + df[kmer]

    df = df.drop(to_drop, axis=1)
    df = df.div(df.sum(axis=1), axis=0)
    return df

def write_tnf_dataset(group, tnf_table, label_df):
    species = tnf_table['species']
    tnf_table = tnf_table.drop('species', axis=1)
    group.create_dataset('tnf', data=tnf_table.values, dtype=float)
    strtype = h5py.special_dtype(vlen=str)
    group.create_dataset('tetramers', data=tnf_table.columns.values, dtype=strtype)
    group.create_dataset('sequence_ids', data=tnf_table.index.values, dtype=strtype)
    group.create_dataset('embeddings_idx', data=species, dtype=int)
    group.create_dataset('embeddings', data=label_df.values, dtype=float)


def read_tnf_dataset(path, group=None):

    with h5py.File(path, 'r') as f:
        g = f
        if group is not None:
            g = f[group]
        X = g['tnf'][:]
        emb = g['embeddings'][:]
        labels = g['embeddings_idx'][:]
        y = emb[labels]

    return X, y, labels

if __name__ == '__main__':
    import sys
    import argparse
    from ..utils import get_accession
    from ..response.embedding import read_embedding
    from ..response.gtdb import check_accessions

    parser = argparse.ArgumentParser()
    parser.add_argument('fof', type=str, help='file of files containing sequences to count tetramers for')
    parser.add_argument('embedding_h5', type=str, help='file that contains embeddings')
    parser.add_argument('output_h5', type=str, help='file to write kmer counts to')

    args = parser.parse_args()

    accessions = list()
    tnf_tables = list()
    species = list()

    fof = open(args.fof, 'r')
    for i, path in enumerate(fof):
        path = path[:-1]
        accessions.append(get_accession(path))
        tnf = get_tetramers(path)
        species.extend([i]*tnf.shape[0])
        tnf_tables.append(tnf)

    tnf_df = pd.concat(tnf_tables, sort=False)
    tnf_df.fillna(0.0, inplace=True)
    tnf_df['species'] = species


    emb, taxa = read_embedding(args.embedding_h5)
    taxa = check_accessions(taxa, trim_src_tag=True)
    colnames = ["dim%02g" % i for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(data=emb, columns=colnames, index=taxa)
    emb_df = emb_df.filter(items=accessions, axis=0)

    with h5py.File(args.output_h5, 'w') as f:
        write_tnf_dataset(f, tnf_df, emb_df)
