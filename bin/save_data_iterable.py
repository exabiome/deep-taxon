import os
import numpy as np
import argparse

from hdmf.common import get_hdf5io
from hdmf.data_utils import DataChunkIterator

from exabiome.sequence.convert import SeqIterator
from exabiome.sequence.dna_table import DNATable

parser = argparse.ArgumentParser()
parser.add_argument('fof', type=str, help='file of Fasta files')
parser.add_argument('out', type=str, help='output HDF5')

args = parser.parse_args()

with open(args.fof, 'r') as f:
    fnapaths = [l[:-1] for l in f.readlines()]

h5path =  args.out

print("reading %d Fasta files" % len(fnapaths))
print("Total size:", sum(os.path.getsize(f) for f in fnapaths))


seqit = SeqIterator(fnapaths, verbose=True)

packed = DataChunkIterator.from_iterable(iter(seqit), maxshape=(None,), buffer_size=2**15, dtype=np.dtype('uint8'))
seqindex = DataChunkIterator.from_iterable(seqit.index_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('int'))
names = DataChunkIterator.from_iterable(seqit.names_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('U'))
ids = DataChunkIterator.from_iterable(seqit.id_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('int'))
taxa = DataChunkIterator.from_iterable(seqit.taxon_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('uint16'))


with get_hdf5io(h5path, 'w') as io:
    table = DNATable('root', 'a test table',
                     io.set_dataio(names,    compression='gzip', chunks=(2**15,)),
                     io.set_dataio(packed,   compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     io.set_dataio(seqindex, compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     io.set_dataio(taxa, compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     id=io.set_dataio(ids, compression='gzip', maxshape=(None,), chunks=(2**15,)))
    io.write(table, exhaust_dci=False)

print("reading %s" % (h5path))
h5size = os.path.getsize(h5path)
print("HDF5 size:", h5size)
