import numpy as np

from hdmf.common import DynamicTableRegion, EnumData, VectorIndex


def load_elements(table):
    for colname in table.colnames:
        col = table[colname]
        if isinstance(col, VectorIndex):
            col = col.target
        if isinstance(col, EnumData):
            col.elements.transform(np.asarray)
        elif isinstance(col, DynamicTableRegion):
            load_elements(col.table)

def check_sequences(argv=None):

    import argparse
    from ..utils import parse_seed, check_argv
    from ..nn.loader import read_dataset
    from ..utils import get_genomic_path
    import skbio.io
    import numpy as np

    argv = check_argv(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('fadir', type=str, help='directory with NCBI sequence files')
    parser.add_argument('-s', '--seed', type=parse_seed, default='', help='seed to use for train-test split')
    parser.add_argument('-n', '--num_seqs', type=int, default=100, help='the number of sequences to check')

    args = parser.parse_args(args=argv)

    dataset, io = read_dataset(args.input)
    difile = dataset.difile

    rand = np.random.RandomState(args.seed)
    if len(difile.seq_table) < args.num_seqs:
        indices = np.arange(len(difile.seq_table))
    else:
        indices = rand.permutation(np.arange(len(difile.seq_table)))[:args.num_seqs]

    indices = np.sort(indices)


    load_elements(difile.seq_table)

    breakpoint()

    seqs = difile.seq_table[indices].sort_values('genome_taxon_id')

    taxon_ids = np.unique(seqs['genome_taxon_id'])
    bad_seqs = list()
    for tid in taxon_ids:
        path = get_genomic_path(tid, args.fadir)
        subdf = seqs[seqs['genome_taxon_id'] == tid]
        for seq in skbio.io.read(path, format='fasta'):
            mask = subdf['sequence_name'] == seq.metadata['id']
            if mask.any():
                if not np.array_equal(subdf['sequence'][mask].iloc[0], seq.values.astype('U')):
                    bad_seqs.append((tid, seq.metadata['id']))

    if len(bad_seqs) > 0:
        print('the following', len(bad_seqs), 'sequences do not match')
        for tid, seqname in bad_seqs:
            print(tid, seqname)
    else:
        print('all sampled sequences match')

