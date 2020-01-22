import pandas as pd

def read_taxonomy(path, leaf_names=None):
    taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    def func(row):
        dat = dict(zip(taxlevels, row['gtdb_taxonomy'].split(';')))
        dat['species'] = dat['species'].split(' ')[1]
        dat['accession'] = row['accession']
        return pd.Series(data=dat)
    taxdf = pd.read_csv(path, header=0, sep='\t')[['accession', 'gtdb_taxonomy']]\
                        .apply(func, axis=1)\
                        .set_index('accession')
    if leaf_names is not None:
        taxdf = taxdf.filter(items=leaf_names, axis=0)
    return taxdf
