import sys
from datetime import datetime
import numpy as np
import exabiome.sequence
from hdmf.common import get_hdf5io
from skbio.sequence import Protein
import skbio.io

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
parser.add_argument('fof', type=str, help='the HDF5 DeepIndex file')
parser.add_argument('idx', type=int, help='the HDF5 DeepIndex file')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

logger.info('loading data %s' % args.input)


io = get_hdf5io(args.input, 'r')
difile = io.read()
difile.set_raw()
tid = difile.seq_table[args.idx][3][1]
sid = difile.seq_table[args.idx][1]


fofin = open(args.fof, 'r')
for line in map(lambda x: x.strip(), fofin):
    if tid in line:
        fasta_file = line
        break
fofin.close()

print(sid, tid, fasta_file)

for seq in skbio.io.read(fasta_file, constructor=Protein, format='fasta'):
    if sid == seq.metadata['id']:
        print(seq)

