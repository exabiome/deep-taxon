import argparse
import logging
import os
import sys

import pandas as pd
import skbio.io as skio
from tqdm import tqdm

def get_logger():
    logger = logging.getLogger('stderr')
    hdlr = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return logger


taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def tax_acc(argv):
    """Computing accuracy of taxonomic classification across all ranks"""

    parser = argparse.ArgumentParser()
    parser.add_argument('tax_cls', type=str, help='the taxonomic classification file for sequences')
    parser.add_argument('taxonomy', type=str, help='the taxonomy table from the GTNet deploy directory')
    parser.add_argument('-o', '--output', type=str, help='the output file to save results to', default=None)
    parser.add_argument('-s', '--min_score', type=float, help='the minimum score a classification must have', default=None)

    args = parser.parse_args(argv)

    logger = get_logger()

    logger.info(f'Reading taxonomy from {args.taxonomy}')
    taxdf = pd.read_csv(args.taxonomy, index_col='accession')

    ar122 = (taxdf['domain'] == 'd__Archaea').values
    bac120 = (taxdf['domain'] == 'd__Bacteria').values
    logger.info(f' - found {ar122.sum()} Archaea genomes and {bac120.sum()} Bacteria genomes')

    # accession,seq_name,domain,phylum,class,order,family,genus,species
    # GCA_000380905.1,AQYW01000001.1,d__Archaea,p__Nanoarchaeota,c__Nanoarchaeia,o__SCGC-AAA011-G17,f__SCGC-AAA011-G17,g__SCGC-AAA011-G17,s__SCGC-AAA011-G17 sp000402515

    logger.info(f'Reading LCA results from {args.tax_cls}')
    cls_df = pd.read_csv(args.tax_cls)
    df2 = cls_df['ID'].str.split('-', 3, expand=True)
    cls_df['accession'] = df2.iloc[:, 0]
    cls_df = cls_df.drop(['ID', 'filename'], axis=1)

    if args.min_score is not None:
        cls_df = cls_df[cls_df['score'] >= args.min_score]

    '''
    filename,ID,domain,phylum,class,order,family,genus,species,score
    benchmarks/run_gtnet.2844110/all_genomic.fna,GCF_004208935.1-141866-NZ_SGPD01000037.1,d__Bacteria,p__Proteobacteria,c__Gammaproteobacteria,o__Enterobacterales,f__Alteromonadaceae,g__Pseudoalteromonas,s__Pseudoalteromonas sp001469895,1.0
    '''

    taxdf = taxdf.filter(cls_df['accession'], axis=0)

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

        pclfd, acc = get_results(taxdf, cls_df, col)
        results['pclfd'].append(pclfd)
        results['accuracy'].append(acc)


        pclfd, acc = get_results(taxdf, cls_df, col, sub=bac120)
        results['bac_pclfd'].append(pclfd)
        results['bac_accuracy'].append(acc)

        pclfd, acc = get_results(taxdf, cls_df, col, sub=ar122)
        results['ar_pclfd'].append(pclfd)
        results['ar_accuracy'].append(acc)

    df = pd.DataFrame(data=results, index=taxlevels[1:])
    if args.output is not None:
        df.to_csv(args.output)
    print(df)


def main():
    tax_acc(sys.argv[1:])

if __name__ == '__main__':
    main()
