import argparse
import logging
import sys

import math
import numpy as np
import pandas as pd
import scipy.sparse as sps
import skbio.stats.distance as ssd
from skbio.sequence import DNA, Protein
from sklearn.preprocessing import LabelEncoder
from hdmf.common import EnumData, VectorData
import skbio.io
import os
from functools import partial

from scipy.spatial.distance import squareform


def add_branches(node, mat, names):
    name = node.name
    names[node.id] = name
    if len(node.children) == 0:
        return
    if len(node.children) != 2:
        raise ValueError("Non-binary tree! I do not know how to handle this")
    for c in node.children:
        mat[node.id, c.id] = c.length
        add_branches(c, mat, names)


def get_tree_graph(node, rep_taxdf):
    node.to_array()
    n_nodes = 2 * len(list(node.tips())) - 1
    adj = sps.lil_matrix((n_nodes, n_nodes))
    names = np.zeros(n_nodes, dtype='U15')
    add_branches(node, adj, names)

    tids = rep_taxdf.index
    gt_indices = np.ones(len(names)) * -1
    for i, name in enumerate(names):
        if name.startswith('GC'):
            gt_indices[i] = np.where(tids == name)[0][0]

    return adj.tocsr(), gt_indices


def seqlen(path):
    kwargs = {'format': 'fasta', 'constructor': DNA, 'validate': False}
    l = 0
    i = 0
    for seq in skbio.io.read(path, **kwargs):
        l += len(seq)
        i += 1
    return i, l

def readseq(path):
    kwargs = {'format': 'fasta', 'constructor': DNA, 'validate': False}
    return list(skbio.io.read(path, **kwargs))


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

def select_embeddings(ids_to_select, taxa_ids, embeddings):
    id_map = {t[3:]: i for i, t in enumerate(taxa_ids)}
    indices = [id_map[tid] for tid in ids_to_select]
    return embeddings[indices]


def prepare_data(argv=None):
    '''Aggregate sequence data GTDB using a file-of-files'''
    from io import BytesIO
    import tempfile
    import h5py

    from datetime import datetime

    from tqdm import tqdm

    from skbio import TreeNode
    from skbio.sequence import DNA, Protein

    from hdmf.common import get_hdf5io
    from hdmf.data_utils import DataChunkIterator

    from ..utils import get_faa_path, get_fna_path, get_genomic_path
    from deep_taxon.sequence.convert import AASeqIterator, DNASeqIterator, DNAVocabIterator, DNAVocabGeneIterator
    from deep_taxon.sequence.dna_table import AATable, DNATable, SequenceTable, TaxaTable, DeepIndexFile, NewickString, CondensedDistanceMatrix, GenomeTable, TreeGraph

    parser = argparse.ArgumentParser()
    parser.add_argument('fadir', type=str, help='directory with NCBI sequence files. This is the directory that contains the \'all\' directory')
    parser.add_argument('metadata', type=str, help='metadata file from GTDB')
    parser.add_argument('out', type=str, help='output HDF5')
    parser.add_argument('-T', '--tree', type=str, help='a Newick file with a tree of representative taxa', default=None)
    parser.add_argument('-A', '--accessions', type=str, default=None, help='file of the NCBI accessions of the genomes to convert')
    parser.add_argument('-d', '--max_deg', type=float, default=None, help='max number of degenerate characters in protein sequences')
    parser.add_argument('-l', '--min_len', type=float, default=None, help='min length of sequences')
    parser.add_argument('--iter', action='store_true', default=False, help='convert using iterators')
    parser.add_argument('-p', '--num_procs', type=int, default=1, help='the number of processes to use for counting total sequence size')
    parser.add_argument('-L', '--total_seq_len', type=int, default=None, help='the total sequence length')
    parser.add_argument('-t', '--tmpdir', type=str, default=None, help='a temporary directory to store sequences')
    parser.add_argument('-H', '--tmp_h5', type=str, default=None, help='the temporary HDF5 file with with sequences')
    parser.add_argument('-N', '--n_seqs', type=int, default=None, help='the total number of sequences')
    parser.add_argument('--print-accessions', action='store_true', default=False, help='print accessions and exit')
    rep_grp = parser.add_mutually_exclusive_group()
    rep_grp.add_argument('-n', '--nonrep', action='store_true', default=False, help='keep non-representative genomes only. keep both by default')
    rep_grp.add_argument('-r', '--rep', action='store_true', default=False, help='keep representative genomes only. keep both by default')
    parser.add_argument('--nontest', action='store_true', default=False, help='get non-test non-representatives. Ignored when --all is used')
    parser.add_argument('-a', '--all', action='store_true', default=False,
                        help='keep all non-representative genomes. By default, only non-reps with the highest and lowest contig count are kept')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-P', '--protein', action='store_true', default=False, help='get paths for protein files')
    grp.add_argument('-C', '--cds', action='store_true', default=False, help='get paths for CDS files')
    grp.add_argument('-G', '--genomic', action='store_true', default=False, help='get paths for genomic files (default)')
    parser.add_argument('-z', '--gzip', action='store_true', default=False,
                        help='GZip sequence table')
    dep_grp = parser.add_argument_group(title="Legacy options you probably do not need")
    dep_grp.add_argument('-e', '--emb', type=str, help='embedding file', default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args=argv)

    if not os.path.exists(args.fadir):
        print(f"Fasta directory {args.fadir} does not exist", file=sys.stderr)
        exit(1)

    if args.total_seq_len is not None:
        if args.n_seqs is None:
            sys.stderr.write("If using --total_seq_len, you must also use --n_seqs\n")
    if args.n_seqs is not None:
        if args.total_seq_len is None:
            sys.stderr.write("If using --n_seqs, you must also use --total_seq_len\n")


    if not any([args.protein, args.cds, args.genomic]):
        args.genomic = True

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    #############################
    # read and filter taxonomies
    #############################
    logger.info('Reading taxonomies from %s' % args.metadata)
    taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    extra_cols = ['contig_count', 'checkm_completeness']
    def func(row):
        dat = dict(zip(taxlevels, row['gtdb_taxonomy'].split(';')))
        dat['species'] = dat['species'] # .split(' ')[1]
        dat['gtdb_genome_representative'] = row['gtdb_genome_representative'][3:]
        dat['accession'] = row['accession'][3:]
        for k in extra_cols:
            dat[k] = row[k]
        return pd.Series(data=dat)

    taxdf = pd.read_csv(args.metadata, header=0, sep='\t', usecols=['accession', 'gtdb_taxonomy', 'gtdb_genome_representative', 'contig_count', 'checkm_completeness'])\
                        .apply(func, axis=1)

    logger.info('Setting accession as index')

    taxdf = taxdf.set_index('accession')
    dflen = len(taxdf)
    logger.info('Found %d total genomes' % dflen)
    taxdf = taxdf[taxdf['gtdb_genome_representative'].str.contains('GC[A,F]_', regex=True)]   # get rid of genomes that are not at NCBI
    taxdf = taxdf[taxdf.index.str.contains('GC[A,F]_', regex=True)]   # get rid of genomes that are not at NCBI
    logger.info('Discarded %d non-NCBI genomes' % (dflen - len(taxdf)))

    rep_taxdf = taxdf[taxdf.index == taxdf['gtdb_genome_representative']]

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
        dflen = len(taxdf)
        if not args.all:
            groups = taxdf[['gtdb_genome_representative', 'contig_count']].groupby('gtdb_genome_representative')
            min_ctgs = groups.idxmin()['contig_count']
            max_ctgs = groups.idxmax()['contig_count']
            accessions = np.unique(np.concatenate([min_ctgs, max_ctgs]))
            if args.nontest:
                dflen = len(taxdf)
                taxdf = taxdf.drop(accessions, axis=0)
                groups = taxdf[['gtdb_genome_representative', 'contig_count']].groupby('gtdb_genome_representative')
                min_ctgs = groups.idxmin()['contig_count']
                max_ctgs = groups.idxmax()['contig_count']
                accessions = np.unique(np.concatenate([min_ctgs, max_ctgs]))
                taxdf = taxdf.filter(accessions, axis=0)
                logger.info('Discarded %d non-test non-representative genomes' % (dflen - len(taxdf)))
            else:
                taxdf = taxdf.filter(accessions, axis=0)
                logger.info('Discarded %d extra non-representative genomes' % (dflen - len(taxdf)))
    elif args.rep:
        taxdf = taxdf[taxdf.index == taxdf['gtdb_genome_representative']]
        logger.info('Discarded %d non-representative genomes' % (dflen - len(taxdf)))

    dflen = len(taxdf)
    logger.info('%d remaining genomes' % dflen)

    if args.print_accessions:
        logger.info('printing accessions and exiting')
        for tid in taxdf.index.values:
            print(tid)
        exit(0)


    ###############################
    # Arguments for constructing the DeepIndexFile object
    ###############################
    di_kwargs = dict()

    taxa_ids = taxdf.index.values

    # get paths to Fasta Files
    fa_path_func = partial(get_genomic_path, directory=args.fadir)
    if args.cds:
        fa_path_func = partial(get_fna_path, directory=args.fadir)
    elif args.protein:
        fa_path_func = partial(get_faa_path, directory=args.fadir)

    map_func = map
    if args.num_procs > 1:
        logger.info(f'using {args.num_procs} processes to locate Fasta files')
        import multiprocessing as mp
        map_func = mp.Pool(processes=args.num_procs).imap

    logger.info('Locating Fasta files for each taxa')
    fapaths = list(tqdm(map_func(fa_path_func, taxa_ids), total=len(taxa_ids)))

    logger.info('Found Fasta files for all accessions')

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


    logger.info(f'Writing {len(rep_taxdf)} taxa to taxa table')
    tt_args = ['taxa_table', 'a table for storing taxa data', rep_taxdf.index.values]
    tt_kwargs = dict()
    for t in taxlevels[:-1]:
        enc = LabelEncoder().fit(rep_taxdf[t].values)
        _data = enc.transform(rep_taxdf[t].values).astype(np.uint32)
        _vocab = enc.classes_.astype('U')
        logger.info(f'{t} - {len(_vocab)} classes')
        tt_args.append(EnumData(name=t, description=f'label encoded {t}', data=_data, elements=_vocab))
    # we have too many species to store this as VocabData, nor does it save any spaces
    tt_args.append(VectorData(name='species', description=f'Microbial species in the form Genus species', data=rep_taxdf['species'].values))

    if emb is not None:
        tt_kwargs['embedding'] = emb
    #tt_kwargs['rep_taxon_id'] = rep_taxdf['gtdb_genome_representative'].values

    taxa_table = TaxaTable(*tt_args, **tt_kwargs)

    h5path = args.out

    logger.info("reading %d Fasta files" % len(fapaths))
    logger.info("Total size: %d", sum(list(map_func(os.path.getsize,  fapaths))))

    tmp_h5_file = None
    if args.protein:
        vocab_it = AAVocabIterator
        SeqTable = SequenceTable
        skbio_cls = Protein
    else:
        vocab_it = DNAVocabIterator
        SeqTable = DNATable
        skbio_cls = DNA

    vocab = np.array(list(vocab_it.characters()))
    if not args.protein:
        np.testing.assert_array_equal(vocab, list('ACYWSKDVNTGRMHB'))


    if args.tmp_h5 is not None:
        tmp_h5_file = h5py.File(args.tmp_h5, 'r')
        sequence = tmp_h5_file['sequences']
        seqindex = tmp_h5_file['sequences_index']
        n_seqs, total_seq_len = seqindex.shape[0], sequence.shape[0]
        genomes = tmp_h5_file['genomes']
        seqlens = tmp_h5_file['seqlens']
        names = tmp_h5_file['seqnames']
        ids = tmp_h5_file['ids']

        taxa = np.zeros(len(fapaths), dtype=int)

        for genome_i, fa in tqdm(enumerate(fapaths), total=len(fapaths)):
            taxid = taxa_ids[genome_i]
            rep_taxid = taxdf['gtdb_genome_representative'][genome_i]
            taxa[genome_i] = np.where(rep_taxdf.index == rep_taxid)[0][0]
    else:
        if args.total_seq_len is None:
            logger.info('counting total number of sqeuences')
            n_seqs, total_seq_len = np.array(list(zip(*tqdm(map_func(seqlen, fapaths), total=len(fapaths))))).sum(axis=1)
            logger.info(f'found {total_seq_len} bases across {n_seqs} sequences')
        else:
            n_seqs, total_seq_len = args.n_seqs, args.total_seq_len
            logger.info(f'As specified, there are {total_seq_len} bases across {n_seqs} sequences')

        logger.info(f'allocating uint8 array of length {total_seq_len} for sequences')

        if args.tmpdir is not None:
            if not os.path.exists(args.tmpdir):
                os.mkdir(args.tmpdir)
            tmpdir = tempfile.mkdtemp(dir=args.tmpdir)
        else:
            tmpdir = tempfile.mkdtemp()

        comp = 'gzip' if args.gzip else None
        tmp_h5_filename = os.path.join(tmpdir, 'sequences.h5')
        logger.info(f'writing temporary sequence data to {tmp_h5_filename}')
        tmp_h5_file = h5py.File(tmp_h5_filename, 'w')
        sequence = tmp_h5_file.create_dataset('sequences', shape=(total_seq_len,), dtype=np.uint8, compression=comp)
        seqindex = tmp_h5_file.create_dataset('sequences_index', shape=(n_seqs,), dtype=np.uint64, compression=comp)
        genomes = tmp_h5_file.create_dataset('genomes', shape=(n_seqs,), dtype=np.uint64, compression=comp)
        seqlens = tmp_h5_file.create_dataset('seqlens', shape=(n_seqs,), dtype=np.uint64, compression=comp)
        names = tmp_h5_file.create_dataset('seqnames', shape=(n_seqs,), dtype=h5py.special_dtype(vlen=str), compression=comp)

        taxa = np.zeros(len(fapaths), dtype=int)

        seq_i = 0
        b = 0
        for genome_i, fa in tqdm(enumerate(fapaths), total=len(fapaths)):
            kwargs = {'format': 'fasta', 'constructor': skbio_cls, 'validate': False}
            taxid = taxa_ids[genome_i]
            rep_taxid = taxdf['gtdb_genome_representative'][genome_i]
            taxa[genome_i] = np.where(rep_taxdf.index == rep_taxid)[0][0]
            for seq in skbio.io.read(fa, **kwargs):
                enc_seq = vocab_it.encode(seq)
                e = b + len(enc_seq)
                sequence[b:e] = enc_seq
                seqindex[seq_i] = e
                genomes[seq_i] = genome_i
                seqlens[seq_i] = len(enc_seq)
                names[seq_i] = vocab_it.get_seqname(seq)
                b = e
                seq_i += 1
        ids = tmp_h5_file.create_dataset('ids', data=np.arange(n_seqs), dtype=int)
        tmp_h5_file.flush()

    io = get_hdf5io(h5path, 'w')

    print([a['name'] for a in GenomeTable.__init__.__docval__['args']])

    genome_table = GenomeTable('genome_table', 'information about the genome each sequence comes from',
                               taxa_ids, taxa, taxa_table=taxa_table)

    #############################
    # read and trim tree
    #############################
    if args.tree:
        logger.info('Reading tree from %s' % args.tree)
        root = TreeNode.read(args.tree, format='newick')

        logger.info('Found %d tips' % len(list(root.tips())))

        logger.info('Transforming leaf names for shearing')
        for tip in root.tips():
            tip.name = tip.name[3:].replace(' ', '_')

        logger.info('converting tree to Newick string')
        bytes_io = BytesIO()
        root.write(bytes_io, format='newick')
        tree_str = bytes_io.getvalue()
        di_kwargs['tree'] = NewickString('tree', data=tree_str)

        logger.info('generating patristic distance matrix')
        # get distances from tree if they are not provided
        tt_dmat = squareform(root.tip_tip_distances().filter(rep_taxdf.index).data.astype(np.float32))
        di_kwargs['distances'] = CondensedDistanceMatrix('distances', data=tt_dmat)

        adj, gt_indices = get_tree_graph(root, rep_taxdf)
        di_kwargs['tree_graph'] = TreeGraph(data=adj, leaves=gt_indices, table=genome_table, name='tree_graph')


    if args.gzip:
        names = io.set_dataio(names,    compression='gzip', chunks=True)
        sequence = io.set_dataio(sequence,   compression='gzip', maxshape=(None,), chunks=True)
        seqindex = io.set_dataio(seqindex, compression='gzip', maxshape=(None,), chunks=True)
        seqlens = io.set_dataio(seqlens, compression='gzip', maxshape=(None,), chunks=True)
        genomes = io.set_dataio(genomes, compression='gzip', maxshape=(None,), chunks=True)
        ids = io.set_dataio(ids, compression='gzip', maxshape=(None,), chunks=True)

    seq_table = SeqTable('seq_table', 'a table storing sequences for computing sequence embedding',
                         names, sequence, seqindex, seqlens, genomes,
                         genome_table=genome_table,
                         id=ids,
                         vocab=vocab)



    difile = DeepIndexFile(seq_table, taxa_table, genome_table, **di_kwargs)

    before = datetime.now()
    io.write(difile, exhaust_dci=False, link_data=False)
    io.close()
    after = datetime.now()
    delta = (after - before).total_seconds()

    logger.info(f'Sequence totals {sequence.dtype.itemsize * sequence.size} bytes')
    logger.info(f'Took {delta} seconds to write after read')

    if tmp_h5_file is not None:
        tmp_h5_file.close()

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


def merge_metadata(argv=None):
    """Merge metadata file from different sources"""

    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='the path to write merged metadata CSV to')
    parser.add_argument('-g', '--gtdb', type=str, nargs='*', help='metadata files from GTDB', default=list())
    parser.add_argument('-i', '--ictv', type=str, nargs='*', help='metadata files from ICTV -- not implemented yet', default=list())

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args=argv)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    df = None

    if args.gtdb:
        dfs = list()
        for csv in args.gtdb:
            logger.info(f'reading {csv}')
            dfs.append(pd.read_csv(csv, header=0, sep='\t'))
        df = pd.concat(dfs)

    df.to_csv(args.output, sep='\t', index=False)



if __name__ == '__main__':
    prepare_data()

