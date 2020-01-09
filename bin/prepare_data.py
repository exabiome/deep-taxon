import os
import io
import sys
import argparse
import logging
import math
import h5py
import numpy as np
import pandas as pd

from skbio import TreeNode

from hdmf.common import get_hdf5io
from hdmf.data_utils import DataChunkIterator

from exabiome.sequence.convert import AASeqIterator, DNASeqIterator
from exabiome.sequence.dna_table import AATable, DNATable, TaxaTable, DeepIndexFile, NewickString, CondensedDistanceMatrix


def select_distances(ids_to_select, taxa_ids, distances):
    id_map = {t[3:]: i for i, t in enumerate(taxa_ids)}
    indices = np.array([id_map[tid] for tid in ids_to_select])
    idx = np.zeros(len(indices)*(len(indices)-1)//2, dtype=int)
    k = 0
    n = int(1/2 + math.sqrt(1/4 + 2 * len(distances)))
    for s_i, _i in enumerate(indices):
        for s_j, _j in enumerate(indices[s_i+1:]):
            if _i > _j:
                i, j = _j, _i
            else:
                i, j = _i, _j
            idx[k] = i*n - ((i+1)*(i+2)//2) + j
            k += 1
    return distances[idx]


def select_embeddings(ids_to_select, taxa_ids, embeddings):
    id_map = {t[3:]: i for i, t in enumerate(taxa_ids)}
    indices = [id_map[tid] for tid in ids_to_select]
    return embeddings[indices]


def get_taxa_id(path):
    c, n = os.path.basename(path).split('_')[0:2]
    return c + '_' + n

parser = argparse.ArgumentParser()
parser.add_argument('fof', type=str, help='file of Fasta files')
parser.add_argument('emb_h5', type=str, help='embedding file')
parser.add_argument('dist_h5', type=str, help='the distances file')
parser.add_argument('tree', type=str, help='the distances file')
parser.add_argument('metadata', type=str, help='metadata file from GTDB')
parser.add_argument('out', type=str, help='output HDF5')
parser.add_argument('-a', '--faa', action='store_true', default=False, help='input is amino acids')
parser.add_argument('-d', '--max_deg', type=float, default=None, help='max number of degenerate characters in protein sequences')
parser.add_argument('-l', '--min_len', type=float, default=None, help='min length of sequences')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

logger.info('reading Fasta paths from %s' % args.tree)
with open(args.fof, 'r') as f:
    fapaths = [l[:-1] for l in f.readlines()]

taxa_ids = list(map(get_taxa_id, fapaths))

#############################
# read and filter distances
#############################
logger.info('reading distances from %s' % args.dist_h5)
with h5py.File(args.dist_h5, 'r') as f:
    dist = f['distances'][:]
    dist_taxa = f['leaf_names'][:].astype('U')
logger.info('selecting distances for taxa found in %s' % args.fof)
dist = select_distances(taxa_ids, dist_taxa, dist)
dist = CondensedDistanceMatrix('distances', data=dist)


#############################
# read and filter taxonomies
#############################
logger.info('reading taxonomies from %s' % args.metadata)
taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
def func(row):
    dat = dict(zip(taxlevels, row['gtdb_taxonomy'].split(';')))
    dat['species'] = dat['species'].split(' ')[1]
    dat['accession'] = row['accession'][3:]
    return pd.Series(data=dat)

logger.info('selecting GTDB taxonomy for taxa found in %s' % args.fof)
taxdf = pd.read_csv(args.metadata, header=0, sep='\t')[['accession', 'gtdb_taxonomy']]\
                    .apply(func, axis=1)\
                    .set_index('accession')\
                    .filter(items=taxa_ids, axis=0)


#############################
# read and filter embeddings
#############################
logger.info('reading embeddings from %s' % args.emb_h5)
with h5py.File(args.emb_h5, 'r') as f:
    emb = f['embedding'][:]
    emb_taxa = f['leaf_names'][:]
logger.info('selecting embeddings for taxa found in %s' % args.fof)
emb = select_embeddings(taxa_ids, emb_taxa, emb)

#############################
# read and trim tree
#############################
logger.info('reading tree from %s' % args.tree)
tree = TreeNode.read(args.tree, format='newick')

logger.info('transforming leaf names for shearing')
for tip in tree.tips():
    tip.name = tip.name[3:].replace(' ', '_')

logger.info('shearing taxa not found in %s' % args.fof)
tree = tree.shear(taxa_ids)

logger.info('converting tree to Newick string')
bytes_io = io.BytesIO()
tree.write(bytes_io, format='newick')
tree_str = bytes_io.getvalue()
tree = NewickString('tree', data=tree_str)


h5path = args.out

logger.info("reading %d Fasta files" % len(fapaths))
logger.info("Total size: %d", sum(os.path.getsize(f) for f in fapaths))

if args.faa:
    logger.info("reading and writing protein sequences")
    seqit = AASeqIterator(fapaths, logger=logger, max_degenerate=args.max_deg, min_seq_len=args.min_len)
    SeqTable = AATable
else:
    logger.info("reading and writing DNA sequences")
    seqit = DNASeqIterator(fapaths, logger=logger, min_seq_len=args.min_len)
    SeqTable = DNATable

packed = DataChunkIterator.from_iterable(iter(seqit), maxshape=(None,), buffer_size=2**15, dtype=np.dtype('uint8'))
seqindex = DataChunkIterator.from_iterable(seqit.index_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('int'))
names = DataChunkIterator.from_iterable(seqit.names_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('U'))
ids = DataChunkIterator.from_iterable(seqit.id_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('int'))
taxa = DataChunkIterator.from_iterable(seqit.taxon_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('uint16'))
seqlens = DataChunkIterator.from_iterable(seqit.seqlens_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('uint32'))

io = get_hdf5io(h5path, 'w')

tt_args = ['taxa_table', 'a table for storing taxa data', taxa_ids, emb]
for t in taxlevels[1:]:
    tt_args.append(taxdf[t].values)

taxa_table = TaxaTable(*tt_args)

seq_table = SeqTable('seq_table', 'a table storing sequences for computing sequence embedding',
                     io.set_dataio(names,    compression='gzip', chunks=(2**15,)),
                     io.set_dataio(packed,   compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     io.set_dataio(seqindex, compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     io.set_dataio(seqlens, compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     io.set_dataio(taxa, compression='gzip', maxshape=(None,), chunks=(2**15,)),
                     taxon_table=taxa_table,
                     id=io.set_dataio(ids, compression='gzip', maxshape=(None,), chunks=(2**15,)))

difile = DeepIndexFile(seq_table, taxa_table, dist, tree)

io.write(difile, exhaust_dci=False)
io.close()

logger.info("reading %s" % (h5path))
h5size = os.path.getsize(h5path)
logger.info("HDF5 size: %d", h5size)
