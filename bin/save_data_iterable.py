import os
import numpy as np
import argparse
import h5py

from hdmf.common import get_hdf5io
from hdmf.data_utils import DataChunkIterator

from exabiome.sequence.convert import SeqIterator
from exabiome.sequence.dna_table import DNATable, TaxaTable, DeepIndexFile

def get_taxa_id(path):
    c, n = os.path.basename(path).split('_')[0:2]
    return c + '_' + n

parser = argparse.ArgumentParser()
parser.add_argument('fof', type=str, help='file of Fasta files')
parser.add_argument('emb', type=str, help='embedding file')
parser.add_argument('out', type=str, help='output HDF5')

args = parser.parse_args()

with open(args.fof, 'r') as f:
    fnapaths = [l[:-1] for l in f.readlines()]

taxa_ids = list(map(get_taxa_id, fnapaths))
taxa_id_map = {t: i for i, t in enumerate(taxa_ids)}
print(taxa_id_map)


to_get = list()
target_indices = list()

emb_file = h5py.File(args.emb, 'r')
try:
    for i, val in enumerate(emb_file['leaf_names'][:]):
        v = val[3:]
        if v in taxa_id_map:
            to_get.append(i)
            target_indices.append(taxa_id_map[v])

    target_indices = np.array(target_indices)
    print(target_indices)

    embeddings = np.zeros((len(to_get), emb_file['embedding'].shape[1]))
    embeddings[target_indices] = emb_file['embedding'][to_get]
finally:
    emb_file.close()

for _1, _2 in zip(taxa_ids, embeddings):
    print(_1, _2)

h5path = args.out

print("reading %d Fasta files" % len(fnapaths))
print("Total size:", sum(os.path.getsize(f) for f in fnapaths))

seqit = SeqIterator(fnapaths, verbose=True)

packed = DataChunkIterator.from_iterable(iter(seqit), maxshape=(None,), buffer_size=2**15, dtype=np.dtype('uint8'))
seqindex = DataChunkIterator.from_iterable(seqit.index_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('int'))
names = DataChunkIterator.from_iterable(seqit.names_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('U'))
ids = DataChunkIterator.from_iterable(seqit.id_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('int'))
taxa = DataChunkIterator.from_iterable(seqit.taxon_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('uint16'))

io = get_hdf5io(h5path, 'w')

dna = DNATable('dna_table', 'a table for storing DNA',
               io.set_dataio(names,    compression='gzip', chunks=(2**15,)),
               io.set_dataio(packed,   compression='gzip', maxshape=(None,), chunks=(2**15,)),
               io.set_dataio(seqindex, compression='gzip', maxshape=(None,), chunks=(2**15,)),
               io.set_dataio(taxa, compression='gzip', maxshape=(None,), chunks=(2**15,)),
               id=io.set_dataio(ids, compression='gzip', maxshape=(None,), chunks=(2**15,)))

taxa = TaxaTable('taxa_table', 'a table for storing taxa data', taxa_ids, embeddings)

difile = DeepIndexFile(dna, taxa)

io.write(difile, exhaust_dci=False)
io.close()

print("reading %s" % (h5path))
h5size = os.path.getsize(h5path)
print("HDF5 size:", h5size)
