import os

from hdmf.common import get_hdf5io
from exabiome.sequence.convert import SeqConcat, pack_ohe_dna
from exabiome.sequence.dna_table import DNATable



fnapath = "../deep_index/gtdb/test_data/genomes/all/GCA/000/989/525/GCA_000989525.1_ASM98952v1/GCA_000989525.1_ASM98952v1_cds_from_genomic.fna.gz"
h5path =  "seq.h5"

# ## Read Fasta sequence

print("reading %s" % (fnapath))
fasize = os.path.getsize(fnapath)
print("Fasta size:", fasize)
sc = SeqConcat()
data, seqindex, ltags = sc._read_path(fnapath)

# ## Pack sequence and write to HDF5 file

packed, padded = pack_ohe_dna(data)


with get_hdf5io(h5path, 'w') as io:
    table = DNATable('root', 'a test table',
                     io.set_dataio(ltags, compression='gzip'),
                     io.set_dataio(packed, compression='gzip'),
                     io.set_dataio(seqindex, compression='gzip'))
    io.write(table)

print("reading %s" % (h5path))
h5size = os.path.getsize(h5path)
print("HDF5 size:", h5size)
