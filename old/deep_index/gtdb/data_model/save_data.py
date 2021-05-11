from storage import SeqConcat, pack_ohe_dna
from dna_table import DNATable
from hdmf.backends.hdf5 import HDF5IO, H5DataIO
import hdmf.common as common
import os



fnapath = "../../../../deep_index/gtdb/test_data/genomes/all/GCA/000/989/525/GCA_000989525.1_ASM98952v1/GCA_000989525.1_ASM98952v1_cds_from_genomic.fna.gz"
h5path =  "../../../../deep_index/gtdb/test_data/genomes/all/GCA/000/989/525/GCA_000989525.1_ASM98952v1/seq.h5"

# ## Read Fasta sequence

print("reading %s" % (fnapath))
fasize = os.path.getsize(fnapath)
print("Fasta size:", fasize)
sc = SeqConcat()
data, seqindex, ltags = sc._read_path(fnapath)

# ## Pack sequence and write to HDF5 file

packed, padded = pack_ohe_dna(data)

table = DNATable('root', 'a test table',
                 H5DataIO(ltags, compression='gzip'),
                 H5DataIO(packed, compression='gzip'),
                 H5DataIO(seqindex, compression='gzip'))

with HDF5IO(h5path, 'w', manager=common.get_manager()) as io:
    io.write(table)

print("reading %s" % (h5path))
h5size = os.path.getsize(h5path)
print("HDF5 size:", h5size)
