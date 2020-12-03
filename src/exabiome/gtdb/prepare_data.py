import math
import numpy as np
import skbio.stats.distance as ssd
from skbio.sequence import DNA, Protein
import skbio.io
import os

def seqlen(path):
    kwargs = {'format': 'fasta', 'constructor': DNA, 'validate': False}
    l = 0
    for seq in skbio.io.read(path, **kwargs):
        l += len(seq)
    return l

def duplicate_dmat_samples(dist, dupes):
    """
    Add extra samples to distance matrix
    """
    new_l = dist.shape[0] + len(dupes)
    new_dist = np.zeros_like(dist, shape=(new_l, new_l))
    new_dist[:dist.shape[0], :dist.shape[0]] = dist
    new_slice = np.s_[dist.shape[0]:dist.shape[0]+len(dupes)]
    old_len_slice = np.s_[0:dist.shape[0]]
    new_dist[new_slice, old_len_slice] = dist[dupes]
    new_dist[old_len_slice, new_slice] = dist[dupes].T
    bottom_corner = np.s_[dist.shape[0]:new_dist.shape[0]]
    new_dist[bottom_corner, bottom_corner] = dist[dupes][:,dupes]
    return new_dist

def get_nonrep_matrix(tids, rep_ids, dist):
    """
    Get a distance matrix for non-representative species using
    a distance matrix for representative species

    Args:
        tids (array-like)       : taxon IDs for taxa to orient matrix to
        rep_ids (array-like)    : taxon IDs for the representative taxa in *tids*
        dist (DistanceMatrix)   : the distance matrix to orient
    """
    orig_dist = dist
    uniq, counts = np.unique(rep_ids, return_counts=True)
    dist = orig_dist.filter(uniq).data
    extra = counts - 1
    indices = np.where(extra > 0)[0]
    dupes = np.repeat(np.arange(len(uniq)), extra)
    rep_map = dict()
    for rep, const in zip(rep_ids, tids):
        rep_map.setdefault(rep, list()).append(const)
    rep_order = np.concatenate([np.arange(dist.shape[0]), dupes])
    new_tids = [ rep_map[uniq[i]].pop() for i in rep_order ]
    dupe_dist = duplicate_dmat_samples(dist, dupes)
    ret = ssd.DistanceMatrix(dupe_dist, ids=new_tids)
    ret = ret.filter(tids)
    return ret

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


def prepare_data(argv=None):
    '''Aggregate sequence data GTDB using a file-of-files'''
    import argparse
    import io
    import sys
    import logging
    import h5py
    import pandas as pd

    from datetime import datetime

    from tqdm import tqdm

    from skbio import TreeNode
    from skbio.sequence import DNA, Protein

    from hdmf.common import get_hdf5io
    from hdmf.data_utils import DataChunkIterator

    from ..utils import get_faa_path, get_fna_path, get_genomic_path
    from exabiome.sequence.convert import AASeqIterator, DNASeqIterator, DNAVocabIterator
    from exabiome.sequence.dna_table import AATable, DNATable, SequenceTable, TaxaTable, DeepIndexFile, NewickString, CondensedDistanceMatrix

    parser = argparse.ArgumentParser()
    parser.add_argument('fadir', type=str, help='directory with NCBI sequence files')
    parser.add_argument('metadata', type=str, help='metadata file from GTDB')
    parser.add_argument('tree', type=str, help='the distances file')
    parser.add_argument('out', type=str, help='output HDF5')
    parser.add_argument('-a', '--accessions', type=str, default=None, help='file of the NCBI accessions of the genomes to convert')
    parser.add_argument('-d', '--max_deg', type=float, default=None, help='max number of degenerate characters in protein sequences')
    parser.add_argument('-l', '--min_len', type=float, default=None, help='min length of sequences')
    parser.add_argument('--iter', action='store_true', default=False, help='convert using iterators')
    parser.add_argument('-p', '--num_procs', type=int, default=1, help='the number of processes to use for counting total sequence size')
    rep_grp = parser.add_mutually_exclusive_group()
    rep_grp.add_argument('-n', '--nonrep', action='store_true', default=False, help='keep non-representative genomes only. keep both by default')
    rep_grp.add_argument('-r', '--rep', action='store_true', default=False, help='keep representative genomes only. keep both by default')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-P', '--protein', action='store_true', default=False, help='get paths for protein files')
    grp.add_argument('-C', '--cds', action='store_true', default=False, help='get paths for CDS files')
    grp.add_argument('-G', '--genomic', action='store_true', default=False, help='get paths for genomic files (default)')
    dep_grp = parser.add_argument_group(title="Legacy options you probably do not need")
    dep_grp.add_argument('-N', '--non_vocab', action='store_false', dest='vocab', default=True, help='bitpack sequences -- it is unlikely this works')
    dep_grp.add_argument('-e', '--emb', type=str, help='embedding file', default=None)
    dep_grp.add_argument('-D', '--dist_h5', type=str, help='the distances file', default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args=argv)

    if not any([args.protein, args.cds, args.genomic]):
        args.genomic = True

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    #############################
    # read and filter taxonomies
    #############################
    logger.info('Reading taxonomies from %s' % args.metadata)
    taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    def func(row):
        dat = dict(zip(taxlevels, row['gtdb_taxonomy'].split(';')))
        dat['species'] = dat['species'].split(' ')[1]
        dat['gtdb_genome_representative'] = row['gtdb_genome_representative'][3:]
        dat['accession'] = row['accession'][3:]
        return pd.Series(data=dat)

    taxdf = pd.read_csv(args.metadata, header=0, sep='\t')[['accession', 'gtdb_taxonomy', 'gtdb_genome_representative']]\
                        .apply(func, axis=1)

    taxdf = taxdf.set_index('accession')
    dflen = len(taxdf)
    logger.info('Found %d total genomes' % dflen)
    taxdf = taxdf[taxdf['gtdb_genome_representative'].str.contains('GC[A,F]_', regex=True)]   # get rid of genomes that are not at NCBI
    taxdf = taxdf[taxdf.index.str.contains('GC[A,F]_', regex=True)]   # get rid of genomes that are not at NCBI
    logger.info('Discarded %d non-NCBI genomes' % (dflen - len(taxdf)))

    if args.accessions is not None:
        logger.info('reading accessions %s' % args.accessions)
        with open(args.accessions, 'r') as f:
            accessions = [l[:-1] for l in f.readlines()]
        dflen = len(taxdf)
        taxdf = taxdf[taxdf.index.isin(accessions)]
        logger.info('Discarded %d genomes not found in %s' % (dflen - len(taxdf), args.accessions))

    dflen = len(taxdf)
    if args.nonrep:
        taxdf = taxdf[taxdf.index != taxdf['gtdb_genome_representative']]
        logger.info('Discarded %d representative genomes' % (dflen - len(taxdf)))
    elif args.rep:
        taxdf = taxdf[taxdf.index == taxdf['gtdb_genome_representative']]
        logger.info('Discarded %d non-representative genomes' % (dflen - len(taxdf)))

    dflen = len(taxdf)
    logger.info('%d remaining genomes' % dflen)

    #############################
    # read and trim tree
    #############################
    logger.info('Reading tree from %s' % args.tree)
    root = TreeNode.read(args.tree, format='newick')

    logger.info('Found %d tips' % len(list(root.tips())))

    logger.info('Transforming leaf names for shearing')
    for tip in root.tips():
        tip.name = tip.name[3:].replace(' ', '_')


    logger.info('Shearing filtered accession from tree')
    rep_ids = taxdf['gtdb_genome_representative'].values
    for rid in rep_ids:
        print(rid)
    root = root.shear(rep_ids)
    logger.info('%d tips remaning' % len(list(root.tips())))

    taxa_ids = taxdf.index.values

    # get paths to Fasta Files
    fa_path_func = get_genomic_path
    if args.cds:
        fa_path_func = get_fna_path
    elif args.protein:
        fa_path_func = get_faa_path
    fapaths = [fa_path_func(acc, args.fadir) for acc in taxa_ids]

    logger.info('Found Fasta files for all accessions')

    ###############################
    # Arguments for constructing the DeepIndexFile object
    ###############################
    di_kwargs = dict()

    # if a distance matrix file has been given, read and select relevant distances
    if args.dist_h5:
        #############################
        # read and filter distances
        #############################
        logger.info('reading distances from %s' % args.dist_h5)
        with h5py.File(args.dist_h5, 'r') as f:
            dist = f['distances'][:]
            dist_taxa = f['leaf_names'][:].astype('U')
        logger.info('selecting distances for taxa found in %s' % args.accessions)
        dist = select_distances(taxa_ids, dist_taxa, dist)
        dist = CondensedDistanceMatrix('distances', data=dist)
        di_kwargs['distances'] = dist


    #############################
    # read and filter embeddings
    #############################
    emb = None
    if args.emb is not None:
        logger.info('reading embeddings from %s' % args.emb)
        with h5py.File(args.emb, 'r') as f:
            emb = f['embedding'][:]
            emb_taxa = f['leaf_names'][:]
        logger.info('selecting embeddings for taxa found in %s' % args.accessions)
        emb = select_embeddings(taxa_ids, emb_taxa, emb)


    logger.info('converting tree to Newick string')
    bytes_io = io.BytesIO()
    root.write(bytes_io, format='newick')
    tree_str = bytes_io.getvalue()
    tree = NewickString('tree', data=tree_str)

    # get distances from tree if they are not provided
    if di_kwargs.get('distances') is None:
        from scipy.spatial.distance import squareform
        tt_dmat = root.tip_tip_distances()
        if (rep_ids != taxa_ids).any():
            tt_dmat = get_nonrep_matrix(taxa_ids, rep_ids, tt_dmat)
        dmat = tt_dmat.data
        di_kwargs['distances'] = CondensedDistanceMatrix('distances', data=dmat)

    h5path = args.out

    logger.info("reading %d Fasta files" % len(fapaths))
    logger.info("Total size: %d", sum(os.path.getsize(f) for f in fapaths))

    vocab = None
    if args.iter:
        if args.vocab:
            if args.protein:
                SeqTable = SequenceTable
                seqit = AAVocabIterator(fapaths, logger=logger, min_seq_len=args.min_len)
            else:
                SeqTable = DNATable
                seqit = DNAVocabIterator(fapaths, logger=logger, min_seq_len=args.min_len)
            vocab = np.array(list(seqit.characters()))
        else:
            if args.protein:
                logger.info("reading and writing protein sequences")
                seqit = AASeqIterator(fapaths, logger=logger, max_degenerate=args.max_deg, min_seq_len=args.min_len)
                SeqTable = AATable
            else:
                logger.info("reading and writing DNA sequences")
                seqit = DNASeqIterator(fapaths, logger=logger, min_seq_len=args.min_len)
                SeqTable = DNATable

        seqit_bsize = 2**25
        if args.protein:
            seqit_bsize = 2**15
        elif args.cds:
            seqit_bsize = 2**18

        # set up DataChunkIterators
        sequence = DataChunkIterator.from_iterable(iter(seqit), maxshape=(None,), buffer_size=seqit_bsize, dtype=np.dtype('uint8'))
        seqindex = DataChunkIterator.from_iterable(seqit.index_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('int'))
        names = DataChunkIterator.from_iterable(seqit.names_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('U'))
        ids = DataChunkIterator.from_iterable(seqit.id_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('int'))
        taxa = DataChunkIterator.from_iterable(seqit.taxon_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('uint16'))
        seqlens = DataChunkIterator.from_iterable(seqit.seqlens_iter, maxshape=(None,), buffer_size=2**0, dtype=np.dtype('uint32'))
    else:
        if args.vocab:
            if args.protein:
                vocab_it = AAVocabIterator
                SeqTable = SequenceTable
                skbio_cls = Protein
            else:
                vocab_it = DNAVocabIterator
                SeqTable = DNATable
                skbio_cls = DNA

            vocab = np.array(list(vocab_it.characters()))

            sequence = list()
            names = list()
            taxa = list()
            seqlens = list()

            logger.info('counting total number of sqeuences')

            map_func = map
            if args.num_procs > 1:
                logger.info(f'using {args.num_procs} to count sequences')
                import multiprocessing as mp
                map_func = mp.Pool(processes=args.num_procs).imap

            lengths = list(map_func(seqlen, fapaths))
            total_seq_len = sum(lengths)
            del map_func
            logger.info(f'found {total_seq_len} bases across {len(lengths)} genomes')

            b = 0
            sequence = np.zeros(total_seq_len, dtype=np.uint8)
            seqlens = list()
            for taxa_i, fa in tqdm(list(enumerate(fapaths))):
                kwargs = {'format': 'fasta', 'constructor': skbio_cls, 'validate': False}
                for seq in skbio.io.read(fa, **kwargs):
                    enc_seq = vocab_it.encode(seq)
                    seqlens.append(len(enc_seq))
                    e = b + seqlens[-1]
                    sequence[b:e] = enc_seq
                    b = e
                    names.append(vocab_it.get_seqname(seq))
                    taxa.append(taxa_i)
            #seqlens = np.array([len(s) for s in sequence])
            seqindex = np.cumsum(seqlens).astype(int)
            #sequence = np.concatenate(sequence)
            ids = np.arange(len(seqlens))
        else:
            raise NotImplementedError('cannot load all data into memory unless using vocab encoding')

    io = get_hdf5io(h5path, 'w')

    tt_args = ['taxa_table', 'a table for storing taxa data', taxa_ids]
    tt_kwargs = dict()
    for t in taxlevels[1:]:
        tt_args.append(taxdf[t].values)
    if emb is not None:
        tt_kwargs['embedding'] = emb
    tt_kwargs['rep_taxon_id'] = rep_ids

    taxa_table = TaxaTable(*tt_args, **tt_kwargs)

    seq_table = SeqTable('seq_table', 'a table storing sequences for computing sequence embedding',
                         io.set_dataio(names,    compression='gzip', chunks=True),
                         io.set_dataio(sequence,   compression='gzip', maxshape=(None,), chunks=True),
                         io.set_dataio(seqindex, compression='gzip', maxshape=(None,), chunks=True),
                         io.set_dataio(seqlens, compression='gzip', maxshape=(None,), chunks=True),
                         io.set_dataio(taxa, compression='gzip', maxshape=(None,), chunks=True),
                         taxon_table=taxa_table,
                         id=io.set_dataio(ids, compression='gzip', maxshape=(None,), chunks=True),
                         vocab=vocab)

    difile = DeepIndexFile(seq_table, taxa_table, tree, **di_kwargs)

    logger.info(f'Sequence totals {sequence.nbytes} bytes')
    before = datetime.now()
    io.write(difile, exhaust_dci=False)
    io.close()
    after = datetime.now()
    delta = (after - before).total_seconds()
    logger.info(f'Took {delta} seconds to write after read')

    logger.info("reading %s" % (h5path))
    h5size = os.path.getsize(h5path)
    logger.info("HDF5 size: %d", h5size)

def count_sequence(argv=None):
    """Count the length of total sequence length for a set of accessions"""
    import argparse
    import sys
    import logging
    import skbio.io
    from skbio.sequence import DNA, Protein
    from ..utils import get_faa_path, get_fna_path, get_genomic_path

    parser = argparse.ArgumentParser()
    parser.add_argument('accessions', type=str, help='file of the NCBI accessions of the genomes to convert')
    parser.add_argument('fadir', type=str, help='directory with NCBI sequence files')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-P', '--protein', action='store_true', default=False, help='get paths for protein files')
    grp.add_argument('-C', '--cds', action='store_true', default=False, help='get paths for CDS files')
    grp.add_argument('-G', '--genomic', action='store_true', default=False, help='get paths for genomic files (default)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args=argv)

    # read accessions
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.info('reading accessions %s' % args.accessions)

    with open(args.accessions, 'r') as f:
        taxa_ids = [l[:-1] for l in f.readlines()]

    # get paths to Fasta Files
    fa_path_func = get_genomic_path
    if args.cds:
        fa_path_func = get_fna_path
    elif args.protein:
        fa_path_func = get_faa_path
    fapaths = [fa_path_func(acc, args.fadir) for acc in taxa_ids]

    kwargs = {'format': 'fasta', 'constructor': DNA, 'validate': False}
    total = 0
    for path in fapaths:
        size = 0
        for seq_i, seq in enumerate(skbio.io.read(path, **kwargs)):
            size += len(seq)
        logger.info(f'{size} - {path}')
        total += size
    logger.info(f'{total} total bases')

if __name__ == '__main__':
    prepare_data()

