import abc
import pandas as pd
import numpy as np
import os

from deep_taxon.utils import parse_logger

class TaxClfParser(metaclass=abc.ABCMeta):

    LEVELS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    def __init__(self, contigs=False, logger=None):
        self.contigs = contigs
        self.logger = logger if logger is not None else parse_logger('')

    @abc.abstractmethod
    def strip_accession(self, row):
        pass

    @abc.abstractmethod
    def format_taxonomy(self, row):
        pass

    def get_separator(self):
        return ','

    def format_row(self, row):
        row = row.copy()
        accession = self.strip_accession(row)
        ar = accession.split('-')
        row['accession'] = ar[0]
        if self.contigs:
            row['length'] = int(ar[1])
            row['ID'] = ar[2]
        self.format_taxonomy(row)
        return row

    def read(self, csv):
        self.logger.info(f'Reading {csv}')
        return pd.read_csv(csv, sep=self.get_separator()).apply(self.format_row, axis=1).set_index('accession')


class CATParser(TaxClfParser):

    def format_taxonomy(self, row):
        clf = row.pop('classification')
        row.pop('reason')
        row.pop('lineage scores')
        clf = [None] * 7
        lineage = row.pop('lineage')
        if not isinstance(lineage, float):
            for i, _clf in enumerate(lineage.split(';')[1:]):
                clf[i] = _clf
        for lvl, _clf in zip(self.LEVELS, clf):
            row[lvl] = clf

    def strip_accession(self, row):
        if self.contigs:
            return row.pop('# contig')
        else:
            return row.pop('# bin')[:15]

    def get_separator(self):
        return '\t'


class SourmashParser(TaxClfParser):

    def format_taxonomy(self, row):
        row.pop('status')
        row.pop('strain')
        row['domain'] = row.pop('superkingdom')

    def strip_accession(self, row):
        if self.contigs:
            return row.pop('ID')
        else:
            return os.path.basename(row.pop('ID'))[:15]


class GTNetParser(TaxClfParser):

    def format_taxonomy(self, row):
        pass

    def strip_accession(self, row):
        if self.contigs:
            row.pop('file')
            return row.pop('ID')
        else:
            return os.path.basename(row.pop('file'))[:15]


class Analysis(metaclass=abc.ABCMeta):

    def __init__(self, metadata, keep_cols=None):
        self.keep_cols = keep_cols
        self.tax_df = self.read_metadata(metadata)

    @classmethod
    def _func(cls, row):
        dat = dict(zip(TaxClfParser.LEVELS, row['gtdb_taxonomy'].split(';')))
        dat['species'] = dat['species'] # .split(' ')[1]
        dat['gtdb_genome_representative'] = row['gtdb_genome_representative'][3:]
        dat['accession'] = row['accession'][3:]
        return pd.Series(data=dat)

    def read_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, header=0, sep='\t', usecols=self.keep_cols)\
                .apply(self._func, axis=1).set_index('accession')

    def add_classifications(self, lca_df):
        sub_tax_df = self.tax_df.filter(lca_df.index, axis=0)
        self.process_classifications(sub_tax_df, lca_df)

    @abc.abstractmethod
    def process_classifications(self, tax_df, lca_df):
        pass


class ContigAnalysis(Analysis):

    def __init__(self, metadata, bins=None):
        """
        Args:
            metadata (str):             The path the the GTDB metadata file
            bins (array-like):          The bin edges for binning contig lengths when
                                        doing performance analysis
        """
        super().__init__(metadata, keep_cols=['accession', 'gtdb_taxonomy', 'gtdb_genome_representative'])
        if bins is None:
            self.bins = 10**np.linspace(1, 8, 71)
        else:
            self.bins = bins
        self.complete = { lvl: np.zeros(len(self.bins) - 1) for lvl in TaxClfParser.LEVELS }
        self.complete_bases = { lvl: np.zeros(len(self.bins) - 1) for lvl in TaxClfParser.LEVELS }
        self.correct = { lvl: np.zeros(len(self.bins) - 1) for lvl in TaxClfParser.LEVELS }
        self.correct_bases = { lvl: np.zeros(len(self.bins) - 1) for lvl in TaxClfParser.LEVELS }
        self.total = np.zeros(len(self.bins) - 1)
        self.total_bases = np.zeros(len(self.bins) - 1)

    def process_classifications(self, tax_df, lca_df):
        length = lca_df['length']
        ins = np.searchsorted(self.bins, length)
        self.total += np.bincount(ins, minlength=len(self.bins)-1)
        self.total_bases += np.bincount(ins, minlength=len(self.bins)-1, weights=length)
        for col in TaxClfParser.LEVELS:
            complete_mask = lca_df[col].notna()
            ins_comp = ins[complete_mask]
            length_comp = length[complete_mask]

            self.complete[col] += np.bincount(ins_comp, minlength=len(self.bins)-1)
            self.complete_bases[col] += np.bincount(ins_comp, minlength=len(self.bins)-1, weights=length_comp)

            correct_mask = tax_df[col][complete_mask] == lca_df[col][complete_mask]
            ins_corr = ins_comp[correct_mask]
            length_corr = length_comp[correct_mask]

            self.correct[col] = np.bincount(ins_corr, minlength=len(self.bins)-1)
            self.correct_bases[col] = np.bincount(ins_corr, minlength=len(self.bins)-1, weights=length_corr)

    def save(self, file):
        np.savez(file, bins=self.bins,
                 total=self.total, total_bases=self.total_bases,
                 complete=self.complete, complete_bases=self.complete_bases,
                 correct=self.correct, correct_bases=self.correct_bases)


class BinAnalysis(Analysis):

    def __init__(self, metadata, bins=None):
        super().__init__(metadata, keep_cols=['accession', 'gtdb_taxonomy', 'gtdb_genome_representative',
                                              'checkm_completeness', 'checkm_contamination'])

    def process_classifications(self, tax_df, lca_df):
        pass


if __name__ == '__main__':

    import argparse
    import logging
    import multiprocessing as mp
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', type=str, help='GTDB metadata file')
    parser.add_argument('clf', type=str, choices=['cat', 'sourmash', 'gtnet'],
                        help='The classifier type')
    parser.add_argument('output', type=str, help='the output file to save results to')
    parser.add_argument('csv', type=str, nargs='+', help='CSV formatted classification output')
    parser.add_argument('-f', '--fof', action='store_true', help='csv argument is a file of files', default=False)
    parser.add_argument('-c', '--contigs', action='store_true', help='CSVs are contig classifications', default=False)
    parser.add_argument('-p', '--n_procs', type=int, help='The number of processes to use for reading', default=0)

    args = parser.parse_args()

    if args.fof:
        tmp = list()
        with open(args.csv[0], 'r') as f:
            for l in f.readlines():
                tmp.append(l[:-1])
        args.csv = tmp
        del tmp

    if args.n_procs > 0:
        pool = mp.Pool(args.n_procs)
        map_func = pool.imap
        logger = mp.get_logger()
    else:
        map_func = map
        logger = logging.getLogger()

    hdlr = logging.StreamHandler(sys.stderr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.setLevel(logging.INFO)
    logger.addHandler(hdlr)


    if args.clf == 'cat':
        logger.info("Parsing output from CAT")
        tcparser = CATParser(contigs=args.contigs)
    elif args.clf == 'sourmash':
        logger.info("Parsing output from Sourmash")
        tcparser = SourmashParser(contigs=args.contigs)
    else:
        logger.info("Parsing output from GTNet")
        tcparser = GTNetParser(contigs=args.contigs)

    if args.contigs:
        logger.info("Doing contig-based analysis")
        logger.info(f"Loading metadata from {args.metadata}")
        analysis = ContigAnalysis(args.metadata)
    else:
        logger.info("Doing bins-based analysis")
        logger.info(f"Loading metadata from {args.metadata}")
        analysis = BinAnalysis(args.metadata)

    for csv_i, lca_df in enumerate(map_func(tcparser.read, args.csv)):
        logger.info(f'Tallying classification results for {args.csv[csv_i]}')
        analysis.add_classifications(lca_df)

    logger.info(f"Saving results to {args.output}")
    analysis.save(args.output)

    logger.info("Done")
