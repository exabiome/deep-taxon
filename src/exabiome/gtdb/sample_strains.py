from .. import command

@command('sample-nonrep')
def sample_strains(argv=None):
    '''Get test strain genomes

    Using file-of-files and metadata, sample species across classes and genera.
    '''
    import numpy as np
    import pandas as pd
    import argparse
    import logging
    import sys
    from .io import read_tree, read_metadata
    from ..utils import get_faa_path, get_fna_path, get_genomic_path
    from .utils import get_taxa_id


    parser = argparse.ArgumentParser()
    parser.add_argument('accessions', type=str, help='the file-of-file with representatives')
    parser.add_argument('metadata', type=str, default=None, help='the GTDB metadata CSV')
    parser.add_argument('fadir', type=str, nargs='?', default=None, help='the directory with NCBI Fasta files')
    parser.add_argument('-s', '--self', action='store_true', default=False, help='include representative in output')
    parser.add_argument('-f', '--files', action='store_true', default=False, help='accessions is a file of Fasta files')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-n', '--n_strains', type=int, default=1, help='the number of strains to get for each representative')
    grp.add_argument('-A', '--all', action='store_true', default=False, help='get all strains for each representative')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-P', '--protein', action='store_true', default=False, help='get paths for protein files')
    grp.add_argument('-C', '--cds', action='store_true', default=False, help='get paths for CDS files')
    grp.add_argument('-G', '--genomic', action='store_true', default=False, help='get paths for genomic files')

    args = parser.parse_args(args=argv)

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    logger.info('reading Fasta paths from %s' % args.accessions)
    with open(args.accessions, 'r') as f:
        taxa_ids = [l[:-1] for l in f.readlines()]

    if args.files:
        taxa_ids = list(map(get_taxa_id, taxa_ids))

    logger.info('reading GTDB representatives from %s' % args.metadata)
    def func(row):
        dat = dict()
        dat['gtdb_genome_representative'] = row['gtdb_genome_representative'][3:]
        dat['accession'] = row['accession'][3:]
        return pd.Series(data=dat)

    logger.info('selecting GTDB taxonomy for taxa found in %s' % args.accessions)
    taxdf = pd.read_csv(args.metadata, header=0, sep='\t')[['accession', 'gtdb_genome_representative']]\
                        .apply(func, axis=1)\
                        .set_index('accession')\

    mask = taxdf['gtdb_genome_representative'].isin(taxa_ids)
    taxdf = taxdf[mask]
    mask = taxdf.index.str.contains('GC[A,F]_', regex=True)
    taxdf = taxdf[mask]
    accs = list()
    for tid in taxa_ids:
        mask = taxdf['gtdb_genome_representative'] == tid
        mask = np.logical_and(mask, taxdf.index != tid)
        subdf = taxdf[mask]
        if args.all:
            n_strains = subdf.shape[0]
        else:
            n_strains = min(subdf.shape[0], args.n_strains)
        if n_strains == 0:
            logger.info(f'no strains found for {tid}')
        elif n_strains < args.n_strains:
            logger.info(f'only {n_strains} strain{"" if n_strains == 1 else "s"} available for {tid}')
        subdf.iloc[:n_strains]
        strain_accessions = subdf.index[:n_strains]
        accs.extend(strain_accessions)
        if args.self:
            accs.append(tid)

    if args.fadir is not None:
        func = get_genomic_path
        if args.cds:
            func = get_fna_path
        elif args.protein:
            func = get_faa_path
        for acc in accs:
            print(func(acc, args.fadir), file=sys.stdout)
    else:
        for acc in accs:
            print(acc, file=sys.stdout)
