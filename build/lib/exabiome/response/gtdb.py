import pandas as pd

_src_tags = {'RS_', 'GB_'}
def check_accession(acc, trim_src_tag=False):
    if trim_src_tag:
        if acc[:3] in _src_tags:
            acc = acc[3:]
    return acc


def check_accessions(accessions, trim_src_tag=False):
    if isinstance(accessions, str):
        raise ValueError("'accessions' must be array-like")
    return [check_accession(acc, trim_src_tag=trim_src_tag) for acc in accessions]


def read_taxonomy(path, leaf_names=None, trim_src_tag=False):
    taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    def func(row):
        dat = dict(zip(taxlevels, row['gtdb_taxonomy'].split(';')))
        dat['species'] = dat['species'].split(' ')[1]
        dat['accession'] = check_accession(row['accession'], trim_src_tag=trim_src_tag)
        return pd.Series(data=dat)
    taxdf = pd.read_csv(path, header=0, sep='\t')[['accession', 'gtdb_taxonomy']]\
                        .apply(func, axis=1)\
                        .set_index('accession')
    if leaf_names is not None:
        taxdf = taxdf.filter(items=leaf_names, axis=0)
    return taxdf
