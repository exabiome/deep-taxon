from storage import SeqIterator
from dna_table import DNATable
from hdmf.backends.hdf5 import HDF5IO, H5DataIO
import hdmf.common as common
from hdmf.data_utils import DataChunkIterator
import numpy as np

import os

fnapaths = [
    "../../../../deep_index/gtdb/test_data/genomes/all/GCA/000/989/525/GCA_000989525.1_ASM98952v1/GCA_000989525.1_ASM98952v1_cds_from_genomic.fna.gz",
    "../../../../deep_index/gtdb/test_data/genomes/all/GCA/000/435/995/GCA_000435995.1_MGS137/GCA_000435995.1_MGS137_cds_from_genomic.fna.gz"
]


h5path =  "../../../../deep_index/gtdb/test_data/genomes/all/GCA/000/989/525/GCA_000989525.1_ASM98952v1/seq.h5"

print("reading %d Fasta files" % len(fnapaths))
print("Total size:", sum(os.path.getsize(f) for f in fnapaths))


seqit = SeqIterator(fnapaths)

packed = DataChunkIterator.from_iterable(iter(seqit), maxshape=(None,), buffer_size=2**15, dtype=np.dtype('uint8'))
seqindex = DataChunkIterator.from_iterable(seqit.index_iter, maxshape=(None,), buffer_size=2**10, dtype=np.dtype('int'))
names = DataChunkIterator.from_iterable(seqit.names_iter, maxshape=(None,), buffer_size=2**10, dtype=np.dtype('U'))
ids = DataChunkIterator.from_iterable(seqit.id_iter, maxshape=(None,), buffer_size=2**10, dtype=np.dtype('int'))

table = DNATable('root', 'a test table',
                 H5DataIO(names,    compression='gzip', chunks=(2**15,)),
                 H5DataIO(packed,   compression='gzip', maxshape=(None,), chunks=(2**15,)),
                 H5DataIO(seqindex, compression='gzip', maxshape=(None,), chunks=(2**15,)),
                 id=H5DataIO(ids, compression='gzip', maxshape=(None,), chunks=(2**15,)))

with HDF5IO(h5path, 'w', manager=common.get_manager()) as io:
    io.write(table, exhaust_dci=False)

print("reading %s" % (h5path))
h5size = os.path.getsize(h5path)
print("HDF5 size:", h5size)
