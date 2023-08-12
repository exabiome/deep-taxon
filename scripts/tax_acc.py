import abc

class TaxClfParser(metaclass=abc.ABCMeta):

    LEVELS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    def __init__(self, contigs=False):
        self.contigs = contigs

    @abs.abstractmethod
    def strip_accession(self, row):
        pass

    @abs.abstractmethod
    def format_taxonomy(self, row):
        pass

    def format_row(self, row):
        accession = self.strip_accession(row)
        ar = accession.split('-')
        row['accession'] = ar[0]
        if self.contigs:
            row['length'] = int(ar[1])
            row['ID'] = ar[2]
        self.format_taxonomy(row)
        return row


class CATParser(TaxClfParser):

    def format_taxonomy(self, row):
        row.pop('classification')
        row.pop('reason')
        row.pop('lineage scores')
        clf = [None] * 6
        for i, _clf in enumerate(row.pop('lineage').split(';')[1:]):
            clf[i] = _clf
        for lvl, _clf in zip(self.LEVELS, clf):
            row[lvl] = clf

    def strip_accession(row):
        if contigs:
            return row.pop('# contig')
        else:
            return row.pop('# bin')[:15]


class SourmashParser(TaxClfParser):

    def format_taxonomy(self, row):
        row.pop('status')
        row.pop('strain')
        row['domain'] = row.pop('superkingdom')

    def strip_accession(row):
        if contigs:
            return row.pop('ID')
        else:
            return os.path.basename(row.pop('ID'))[:15]


class GTNetParser(TaxClfParser):

    def format_taxonomy(self, row):
        pass

    def strip_accession(row):
        if contigs:
            row.pop('file')
            return row.pop('ID')
        else:
            return os.path.basename(row.pop('file'))[:15]


class Analysis(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def add_classifications(self, tax_df, lca_df):
        pass


class ContigAnalysis(Analysis):

    def add_classifications(self, tax_df, lca_df):
        for col in TaxClfParser.LEVELS[1:]:
            logger.info(f'computing results for {col}')
            mask = lca_df[col].notna()
            results['pclfd'].append(mask.mean())
            length = lca_df['length']
            results['bases_pclfd'].append(length[mask].sum()/length.sum())
            true = sub_tax_df[col][mask]
            pred = lca_df[col][mask]
            results['accuracy'].append(accuracy_score(true, pred))


class BinAnalysis(Analysis):

    def add_classifications(self, tax_df, lca_df):
        pass


def func(row):
    dat = dict(zip(TaxClfParser.LEVELS, row['gtdb_taxonomy'].split(';')))
    dat['species'] = dat['species'] # .split(' ')[1]
    dat['gtdb_genome_representative'] = row['gtdb_genome_representative'][3:]
    dat['accession'] = row['accession'][3:]
    return pd.Series(data=dat)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', type=str, help='GTDB metadata file')
    parser.add_argument('clf', type=str, options=['cat', 'sourmash', 'gtnet'],
                        help='The classifier type')
    parser.add_argument('csv', type=str, nargs='+', help='CSV formatted classification output')
    parser.add_argument('-f', '--fof', action='store_true', help='csv argument is a file of files', default=False)
    parser.add_argument('-c', '--contigs', action='store_true', help='CSVs are contig classifications', default=False)
    parser.add_argument('-o', '--output', type=str, help='the output file to save results to', default=None)

    args = parser.parse_args()

    if args.fof:
        tmp = list()
        with open(args.csv[0], 'r') as f:
            for l in f.readlines():
                tmp.append(l[:-1])
        args.csv = tmp
        del tmp

    logger = parse_logger('')
    logger.info(f'reading GTDB metadata file from {args.metadata}')
    keep_cols = ['accession', 'gtdb_taxonomy', 'gtdb_genome_representative']
    if not args.contigs:
        keep_cols.append('checkm_completeness')
        keep_cols.append('checkm_contamination')
    tax_df = pd.read_csv(args.metadata, header=0, sep='\t', usecols=keep_cols).apply(func, axis=1).set_index('accession')

    sep = ','
    if args.clf == 'cat':
        tcparser = CATParser(contigs=args.contigs)
        sep = '\t'
    elif args.clf == 'sourmash':
        tcparser = SourmashParser(contigs=args.contigs)
    else:
        tcparser = GTNetParser(contigs=args.contigs)

    if args.contigs:
        analysis = ContigAnalysis()
    else:
        analysis = BinAnalysis()

    for csv in args.csv:
        logger.info(f'reading sourmash taxonomic classifications from {csv}')
        lca_df = pd.read_csv(csv, sep=sep).apply(tcparser.format_row, axis=1).set_index('accession')

        sub_tax_df = tax_df.filter(lca_df.index, axis=0)
        analysis.add_classifications(sub_tax_df, lca_df)
