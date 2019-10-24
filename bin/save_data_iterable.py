import os
import numpy as np

from hdmf.common import get_hdf5io
from hdmf.data_utils import DataChunkIterator

from exabiome.sequence.convert import SeqIterator
from exabiome.sequence.dna_table import DNATable


fnapaths = [
    "../deep_index/gtdb/test_data/genomes/all/GCA/000/989/525/GCA_000989525.1_ASM98952v1/GCA_000989525.1_ASM98952v1_cds_from_genomic.fna.gz",
    "../deep_index/gtdb/test_data/genomes/all/GCA/000/435/995/GCA_000435995.1_MGS137/GCA_000435995.1_MGS137_cds_from_genomic.fna.gz"
]


h5path =  "seq.h5"

print("reading %d Fasta files" % len(fnapaths))
print("Total size:", sum(os.path.getsize(f) for f in fnapaths))


seqit = SeqIterator(fnapaths)

packed = DataChunkIterator.from_iterable(iter(seqit), maxshape=(None,), buffer_size=2**15, dtype=np.dtype('uint8'))
seqindex = DataChunkIterator.from_iterable(seqit.index_iter, maxshape=(None,), buffer_size=2**10, dtype=np.dtype('int'))
names = DataChunkIterator.from_iterable(seqit.names_iter, maxshape=(None,), buffer_size=2**10, dtype=np.dtype('U'))
ids = DataChunkIterator.from_iterable(seqit.id_iter, maxshape=(None,), buffer_size=2**10, dtype=np.dtype('int'))


with get_hdf5io(h5path, 'w') as io:
    table = DNATable('root', 'a test table',
                     io.set_dataio(names,    compression='gzip', chunks=(2**15,)),
                     io.set_dataio(packed,   compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     io.set_dataio(seqindex, compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     id=io.set_dataio(ids, compression='gzip', maxshape=(None,), chunks=(2**15,)))
    io.write(table, exhaust_dci=False)

print("reading %s" % (h5path))
h5size = os.path.getsize(h5path)
print("HDF5 size:", h5size)
