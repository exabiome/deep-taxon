import argparse
import logging
import os
import sys

import pandas as pd
from tqdm import tqdm

taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def sorted_groupby(df, *cols):
    dfs = list()
    cur = None
    rows = list()
    keys = list()
    for idx, row in df.iterrows():
        key = tuple(row[c] for c in cols)
        if cur is not None and cur != key:
            dfs.append(pd.DataFrame(rows))
            keys.append(cur)
            rows = list()
        cur = key
        rows.append(row)
    dfs.append(pd.DataFrame(rows))
    keys.append(cur)
    return zip(keys, dfs)


def chunks_apply(dfs, func, *args, **kwargs):
    last_grp = (None, None)
    rows = list()
    for df in dfs:
        for grp in sorted_groupby(df, 'accession', 'seq_name'):
            if last_grp[0] == grp[0]:
                rows.pop()
                grp = (grp[0], pd.concat([last_grp[1], grp[1]]))
            rows.append(func(grp[1], *args, **kwargs))
            last_grp = grp
    return pd.DataFrame(rows)


def get_logger():
    logger = logging.getLogger('stderr')
    hdlr = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return logger


def read_metadata(path):
    extra_cols = ['contig_count', 'checkm_completeness']
    extra_cols = []
    def func(row):
        dat = dict(zip(taxlevels, row['gtdb_taxonomy'].split(';')))
        dat['species'] = dat['species'] # .split(' ')[1]
        dat['gtdb_genome_representative'] = row['gtdb_genome_representative'][3:]
        dat['accession'] = row['accession'][3:]
        for k in extra_cols:
            dat[k] = row[k]
        return pd.Series(data=dat)

    taxdf = pd.read_csv(path, header=0, sep='\t')[['accession', 'gtdb_taxonomy', 'gtdb_genome_representative']]\
                        .apply(func, axis=1).set_index('accession')
    return taxdf


def prepare_metadata(argv):
    '''Merge the GTDB metadata files'''
    parser = argparse.ArgumentParser()
    parser.add_argument('ar122_metadata', type=str, help='ar122 GTDB metadata files')
    parser.add_argument('bac120_metadata', type=str, help='bac120 GTDB metadata files')
    parser.add_argument('output', type=str, help='the CSV to save preprocessed GTDB taxonomy to', default='gtdb.taxonomy.csv')

    args = parser.parse_args(argv)
    logger = get_logger()

    logger.info(f'reading {args.ar122_metadata}')
    ar122_df = read_metadata(args.ar122_metadata)
    logger.info(f'reading {args.bac120_metadata}')
    bac120_df = read_metadata(args.bac120_metadata)
    logger.info(f'concatenating DataFrames')
    taxdf = pd.concat([ar122_df, bac120_df], axis=0)
    logger.info(f'saving taxonomy to {args.output}')
    taxdf.to_csv(args.output)
    logger.info(f'done')


def orf_lca(argv):
    '''Aggregate hits for each ORF to get the LCA for individual ORFS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('daln', type=str, help='diamond output in Blast 6 format')
    parser.add_argument('output', type=str, help='the file to save LCA ORFs output to')
    parser.add_argument('-s', '--bitscore_cutoff', type=float, help='fraction of max bitscore for including hits. default=0.9', default=0.9)
    parser.add_argument('-a', '--ar122_metadata', type=str, help='ar122 GTDB metadata files', default=None)
    parser.add_argument('-b', '--bac120_metadata', type=str, help='bac120 GTDB metadata files', default=None)
    parser.add_argument('-t', '--taxonomy', type=str, help='the preprocessed GTDB taxonomy file. See prep-meta command', default=None)

    args = parser.parse_args(argv)

    logger = get_logger()

    if args.bitscore_cutoff > 1.0 or args.bitscore_cutoff < 0.0:
        print("'cutoff' (-s, --bitscore_cutoff) must be between 0 and 1.0", file=sys.stderr)
        exit(1)

    if args.taxonomy is None and args.ar122_metadata is None and args.bac120_metadata is None:
        print("Please provide metadata files (-a, -b) or a taxonomy CSV (-t)")
        exit(1)

    if args.taxonomy is None:
        ar122_df = read_metadata(args.ar122_metadata)
        bac120_df = read_metadata(args.bac120_metadata)
        taxdf = pd.concat([ar122_df, bac120_df], axis=0)
        path = 'gtdb.taxonomy.csv'
        logger.info(f'saving taxonomy to {path}')
        taxdf.to_csv(path)
    else:
        logger.info(f'reading taxonomy from {args.taxonomy}')
        taxdf = pd.read_csv(args.taxonomy, index_col='accession')

    def split_cols(df):
        # gen_acc, prot_len, seq_acc =
        df2 = df['query'].str.split('-', 2, expand=True)
        # seq_acc, orfid
        df3 = df2.iloc[:, 2].str.rsplit('_', 1, expand=True)
        df4 = df['subject'].str.split('-', 1, expand=True)
        cols = {'accession': df2.iloc[:, 0],
                'seq_name': df3.iloc[:, 0],
                'orf_id': df3.iloc[:, 1],
                'hit': df4.iloc[:, 0],
                'bitscore': df['bitscore'].astype(float)}
        return pd.DataFrame(data=cols)

    def hits_lca(group, taxdf, frac=0.9):
        scores = group['bitscore'].values
        order = scores.argsort()[::-1]
        best_score = scores[order[0]]
        mask = scores >= frac * best_score
        row = group.iloc[mask]

        # compute LCA for each ORF
        tax = taxdf.filter(row['hit'], axis=0).drop('gtdb_genome_representative', axis=1)
        if len(row) == 1:
            tax = tax.iloc[0].to_dict()
            tax['bitscore'] = best_score
        else:
            tax = {k: v[0] if len(set(v)) == 1 else None for k, v in tax.to_dict(orient='list').items()}
            tax['bitscore'] = scores[mask].mean()

        for k in ('accession', 'seq_name', 'orf_id'):
            tax[k] = row[k].values[0]

        return pd.Series(data=tax)

    logger.info(f'reading {args.daln}')
    tfr = pd.read_csv(args.daln, sep='\t', chunksize=50000,
                      usecols=[0, 1, 11],
                      names=['query', 'subject', 'bitscore'],
                      dtype={'query': str, 'subject': str, 'bitscore': float})
    orfs_df = chunks_apply(tqdm(map(split_cols, tfr)), hits_lca, taxdf, frac=args.bitscore_cutoff)

    orfs_df = orfs_df[['accession', 'seq_name', 'orf_id', 'bitscore'] + taxlevels]

    logger.info(f'saving ORF hits to {args.output}')
    orfs_df.to_csv(args.output, index=False)


def agg_orfs(argv):
    '''Aggregate ORF LCAs to get a taxonomic assignment for individual sequences'''
    parser = argparse.ArgumentParser()
    parser.add_argument('orf_lca', type=str, help='LCA ORFs files. See orf-lca', nargs='+')
    parser.add_argument('-o', '--output', type=str, help='the file to save LCA ORFs output to')
    parser.add_argument('-m', '--bsmax_cutoff', type=float, help='fraction of Bmax cutoff for assigning taxa. default=0.5', default=0.5)

    args = parser.parse_args(argv)

    logger = get_logger()

    if args.bsmax_cutoff > 1.0 or args.bsmax_cutoff < 0.0:
        print("'cutoff' (-m, --bsmax_cutoff) must be between 0 and 1.0", file=sys.stderr)
        exit(1)

    def agg_orfs(df, bsmax_cutoff=0.5):
        scores = df['bitscore'].values
        max_score = scores.sum()
        bsmax_cutoff = max_score * bsmax_cutoff
        taxclass = {'accession': df['accession'].values[0], 'seq_name': df['seq_name'].values[0]}
        for lvl in taxlevels:
            # aggregate bitscores for each taxon
            d = dict()
            for tax, score in zip(df[lvl], scores):
                if tax is None:
                    continue
                val = d.setdefault(tax, 0)
                d[tax] = val + score

            # assign taxa with bitscore above bsmax_cutoff
            taxclass[lvl] = None
            for k, v in d.items():
                if v >= bsmax_cutoff:
                    taxclass[lvl] = k
                    break

        return pd.Series(data=taxclass)

    def read(path):
        logger.info(f'reading {path}')
        return pd.read_csv(path)

    agg_df = chunks_apply((read(path) for path in args.orf_lca),
                          agg_orfs, bsmax_cutoff=args.bsmax_cutoff)
    agg_df = agg_df[['accession', 'seq_name'] + taxlevels]
    if args.output is not None:
        logger.info(f'saving sequence LCA to {args.output}')
        agg_df.to_csv(args.output, index=False)
    else:
        agg_df.to_csv(sys.stdout, index=False)

def tax_acc(argv):
    """Computing accuracy of taxonomic classification across all ranks"""

    parser = argparse.ArgumentParser()
    parser.add_argument('lca', type=str, help='aggregated ORF LCA output. See agg-orf command')
    parser.add_argument('taxonomy', type=str, help='the preprocessed GTDB taxonomy file. See prep-meta command')
    parser.add_argument('-o', '--output', type=str, help='the output file to save results to', default=None)

    args = parser.parse_args(argv)

    logger = get_logger()

    # accession,domain,phylum,class,order,family,genus,species,gtdb_genome_representative
    logger.info(f'Reading taxonomy from {args.taxonomy}')
    taxdf = pd.read_csv(args.taxonomy, index_col='accession')

    ar122 = (taxdf['domain'] == 'd__Archaea').values
    bac120 = (taxdf['domain'] == 'd__Bacteria').values
    logger.info(f' - found {ar122.sum()} Archaea genomes and {bac120.sum()} Bacteria genomes')

    # accession,seq_name,domain,phylum,class,order,family,genus,species
    # GCA_000380905.1,AQYW01000001.1,d__Archaea,p__Nanoarchaeota,c__Nanoarchaeia,o__SCGC-AAA011-G17,f__SCGC-AAA011-G17,g__SCGC-AAA011-G17,s__SCGC-AAA011-G17 sp000402515

    logger.info(f'Reading LCA results from {args.lca}')
    lca_df = pd.read_csv(args.lca)

    taxdf = taxdf.filter(lca_df['accession'], axis=0)

    results = {'accuracy': list(), 'pclfd': list(), 'bac_accuracy': list(),
               'bac_pclfd': list(), 'ar_accuracy': list(), 'ar_pclfd': list()}
    ar122 = (taxdf['domain'] == 'd__Archaea').values
    bac120 = (taxdf['domain'] == 'd__Bacteria').values

    logger.info(f' - found {ar122.sum()} Archaea sequences and {bac120.sum()} Bacteria sequences')

    ar122_tax = taxdf.index[ar122]
    bac120_tax = taxdf.index[bac120]
    logger.info(f' - found {len(set(ar122_tax))} Archaea genomes and {len(set(bac120_tax))} Bacteria genomes')

    def get_results(tdf, ldf, col, sub=None):
        if sub is not None:
            tdf = tdf.iloc[sub]
            ldf = ldf.iloc[sub]
        mask = ldf[col].notna().values
        true = tdf[col][mask].values
        pred = ldf[col][mask].values
        eq = true == pred
        return mask.mean(), eq.mean()

    for col in taxlevels[1:]:
        logger.info(f'computing results for {col}')

        pclfd, acc = get_results(taxdf, lca_df, col)
        results['pclfd'].append(pclfd)
        results['accuracy'].append(acc)


        pclfd, acc = get_results(taxdf, lca_df, col, sub=bac120)
        results['bac_pclfd'].append(pclfd)
        results['bac_accuracy'].append(acc)

        pclfd, acc = get_results(taxdf, lca_df, col, sub=ar122)
        results['ar_pclfd'].append(pclfd)
        results['ar_accuracy'].append(acc)

    df = pd.DataFrame(data=results, index=taxlevels[1:])
    if args.output is not None:
        df.to_csv(args.output)
    print(df)


def main():
    cmds = {
        'prep-meta': prepare_metadata,
        'orf-lca': orf_lca,
        'agg-orfs': agg_orfs,
        'tax-acc': tax_acc,
    }
    if len(sys.argv) == 1 or sys.argv[1] not in cmds or sys.argv[1] in ('-h', '--help', 'help'):
        print(f"Usage: python {os.path.basename(sys.argv[0])} [cmd]")
        print("The following commands are available")
        for k, v in cmds.items():
            print(f'    {k:12}  {v.__doc__}')
        print('    help          print this help statement')
    else:
        cmds[sys.argv[1]](sys.argv[2:])

if __name__ == '__main__':
    main()
