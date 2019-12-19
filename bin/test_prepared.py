from datetime import datetime
import io
import sys
import argparse
import logging
import math
import h5py
import numpy as np

import exabiome.sequence.dna_table
from hdmf.common import get_hdf5io
from skbio.sequence import Protein, DNA
import skbio.io

parser = argparse.ArgumentParser()
parser.add_argument('fof', type=str, help='file of Fasta files')
parser.add_argument('h5', type=str, help='HDF5 file to test')
parser.add_argument('-p', '--protein', action='store_true', default=False, help='input is amino acids')
parser.add_argument('-s', '--seed', type=int, default=int(datetime.now().timestamp()) , help='the seed for sampling random sequences')
parser.add_argument('-n', '--n_seqs', type=float, default=100.0, help='the number of sequences to test')
parser.add_argument('-i', '--index', type=int, nargs='+', default=None, help='specific indices to check')

args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stderr))

logger.info('using seed %d' % args.seed)
random = np.random.RandomState(args.seed)

logger.info('opening %s' % args.h5)
hdmfio = get_hdf5io(args.h5, 'r')
difile = hdmfio.read()
n_total_seqs = len(difile.seq_table)
logger.info('found %d sequences' % n_total_seqs)

if args.index is not None:
    idx = set(args.index)
    logger.info('checking sequences %s' % ", ".join(map(str, args.index)))
else:
    n_seqs = math.round(args.n_seqs * n_total_seqs) if args.n_seqs < 1.0 else int(args.n_seqs)
    idx = set(random.permutation(n_total_seqs)[:n_seqs])
    logger.info('sampling %d sequences' % n_seqs)

constructor = Protein if args.protein else DNA
count = 0

try:
    fofin = open(args.fof, 'r')
    for line in fofin:
        fapath = line.strip()
        it = skbio.io.read(fapath, format='fasta', constructor=constructor)
        for seq in it:
            if count in idx:
                row = difile[count]
                sequence = difile.to_sequence(row['sequence'])
                name = row['name']
                if name != seq.metadata['id']:
                    print(count, 'name', sep='\t', file=sys.stdout)
                if sequence != str(seq):
                    print(count, 'sequence', sep='\t', file=sys.stdout)
                    print(sequence)
                    print(str(seq))

            count += 1
finally:
    hdmfio.close()

