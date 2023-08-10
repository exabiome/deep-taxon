import os

from ..utils import ArgumentParser

def get_taxa_id(path):
    c, n = os.path.basename(path).split('_')[0:2]
    return c + '_' + n


def get_accessions(argv=None):
    import pandas as pd
    import numpy as np

    parser = ArgumentParser()
    parser.add_argument('metadata', type=str, help='metadata file from GTDB')
    rep_grp = parser.add_mutually_exclusive_group()
    rep_grp.add_argument('-r', '--rep', action='store_true', default=False, help='keep representative genomes only. keep both by default')
    rep_grp.add_argument('-t', '--nonrep_test', action='store_true', default=False, help='keep test non-representatives')
    rep_grp.add_argument('-c', '--nonrep_calib', action='store_true', default=False, help='keep calibration non-representatives')

    args = parser.parse_args(args=argv)

    taxdf = pd.read_csv(args.metadata, header=0, sep='\t', usecols=['accession', 'gtdb_genome_representative', 'contig_count'])

    taxdf = taxdf.set_index('accession')

    if args.rep:
        accessions = taxdf[taxdf.index == taxdf['gtdb_genome_representative']].index
    elif (args.nonrep_test or args.nonrep_calib):
        nonrep_taxdf = taxdf[taxdf.index != taxdf['gtdb_genome_representative']]

        groups = nonrep_taxdf[['gtdb_genome_representative', 'contig_count']].groupby('gtdb_genome_representative')
        min_ctgs = groups.idxmin()['contig_count']
        max_ctgs = groups.idxmax()['contig_count']
        best_worst_nonrep_accessions = np.unique(np.concatenate([min_ctgs, max_ctgs]))

        tmp_taxdf = nonrep_taxdf.drop(best_worst_nonrep_accessions , axis=0)
        groups = tmp_taxdf[['gtdb_genome_representative', 'contig_count']].groupby('gtdb_genome_representative')
        min_ctgs = groups.idxmin()['contig_count']
        max_ctgs = groups.idxmax()['contig_count']
        accessions = np.unique(np.concatenate([min_ctgs, max_ctgs]))
        if args.nonrep_test:
            accessions = nonrep_taxdf.drop(accessions, axis=0).index
    else:
        accessions = taxdf.index

    for acc in accessions:
        print(acc[3:])


def make_fof(argv=None):
    '''Find files and print paths for accessions'''
    import logging
    import sys
    import glob
    import h5py

    from skbio import TreeNode

    from ..utils import get_faa_path, get_fna_path, get_genomic_path


    epi = "fadir should be organized with the same structure as the NCBI FTP site"
    parser = ArgumentParser(epilog=epi)
    parser.add_argument('fadir', type=str, help='directory with NCBI sequence files. This is the directory that contains the \'all\' directory')
    parser.add_argument('accessions', type=str, help='A file containing accessions')
    parser.add_argument('-t', '--tree', action='store_true', default=False, help='accessions are from a tree in Newick format')
    parser.add_argument('-f', '--hdmf_file', action='store_true', default=False, help='get accessions from a DeepIndex HDMF file')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-P', '--protein', action='store_true', default=False, help='get paths for protein files')
    grp.add_argument('-C', '--cds', action='store_true', default=False, help='get paths for CDS files')
    grp.add_argument('-G', '--genomic', action='store_true', default=False, help='get paths for genomic files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='log to stderr')


    args = parser.parse_args(args=argv)

    if not any([args.protein, args.cds, args.genomic]):
        args.genomic = True

    loglvl = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(stream=sys.stderr, level=loglvl, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    if args.tree:
        #############################
        # read tree
        #############################
        logger.info('reading tree from %s' % args.accessions)
        tree = TreeNode.read(args.accessions, format='newick')

        logger.info('getting taxa IDs from tree')
        tid_set = set()
        for tip in tree.tips():
            name = tip.name
            if name[0:2] in ('RS', 'GB'):
                name = name[3:]
            name = name.replace(' ', '_')
            tid_set.add(name)


        logger.info('getting Fasta paths from %s' % args.fadir)
        suffix = 'protein' if args.protein else 'genomic'
        pattern = "%s/all/G*/[0-9][0-9][0-9]/[0-9][0-9][0-9]/[0-9][0-9][0-9]/*/*%s*.gz" % (args.fadir, suffix)
        for path in glob.iglob(pattern):
            if 'cds' in path and not args.cds:
                continue
            if 'rna' in path:
                continue
            taxa_id = get_taxa_id(path)
            if taxa_id in tid_set:
                print(path, file=sys.stdout)
                tid_set.remove(taxa_id)

        for tid in sorted(tid_set):
            print(tid, file=sys.stderr)

    else:
        accessions = None
        if args.hdmf_file:
            with h5py.File(args.accessions, 'r') as f:
                accessions = f['genome_table']['taxon_id'][:]
        else:
            if os.path.exists(args.accessions):
                with open(args.accessions, 'r') as f:
                    accessions = f.readlines()
            else:
                accessions = [args.accessions]

        func = get_genomic_path
        if args.cds:
            func = get_fna_path
        elif args.protein:
            func = get_faa_path
        for line in accessions:
            print(func(line.strip(), args.fadir), file=sys.stdout)


def split_fof(argv=None):

    import sys
    import multiprocessing as mp

    import numpy as np
    from tqdm import tqdm

    from ..utils import balsplit, path_size

    parser = ArgumentParser()
    parser.add_argument("fof", help="the file of files to split")
    parser.add_argument("n_splits", type=int, help="the number of split")
    parser.add_argument("outbase", type=str, help="the basename for the files")
    parser.add_argument("-p", "--n_procs", type=int, help="the number of processes to calculate file size", default=16)


    args = parser.parse_args(args=argv)
    pool = mp.Pool(args.n_procs)

    print(f"Reading {args.fof} and calculating sizes")
    paths = list()
    sizes = list()
    with open(args.fof, 'r') as f:
        it = pool.imap(path_size, f, chunksize=1000)
        for path, size in tqdm(it):
            paths.append(path)
            sizes.append(size)

    paths = np.array(paths)
    sizes = np.array(sizes)

    print(f"Calculating split")
    indices = balsplit(sizes, args.n_splits)

    print(f"Writing splits")
    for i in tqdm(range(args.n_splits)):
        with open(f"{args.outbase}.{i:04}.fof", "w") as f:
            for path in paths[indices[i]]:
                print(path, file=f)


def append_accessions(argv=None):
    import sys

    from skbio import DNA
    import skbio.io as skio

    desc = "fasta_path must be a file path from NCBI and have the genome assembly accession in it"
    parser = ArgumentParser(description=desc)
    parser.add_argument('fasta_path', type=str, help='the fasta file to append the prefix to')

    args = parser.parse_args(args=argv)
    args.prefix = os.path.basename(args.fasta_path)[:15]

    seqs = list()
    tmp_fa = sys.stdout
    w = 100
    for seq in skio.read(args.fasta_path, format='fasta', constructor=DNA):
        seq.metadata['id'] = args.prefix+"-"+str(len(seq))+"-"+seq.metadata['id']
        seqs.append(seq)
        tmp_fa.write('>')
        tmp_fa.write(seq.metadata['id'])
        tmp_fa.write('\n')
        for s in range(0, len(seq), w):
            tmp_fa.write(''.join(seq.values[s:s+w].astype('U')))
            tmp_fa.write('\n')


if __name__ == '__main__':
    make_fof()
