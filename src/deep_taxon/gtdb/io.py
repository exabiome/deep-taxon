import numpy as np
import pandas as pd
from skbio.tree import TreeNode

def read_tree(tree_path):
    """
    Read Newick formatted tree from GTDB.

    Only tips that have a NCBI accession will be kept.
    i.e. GTDB MAGs are pruned from the tree
    """
    tree = TreeNode.read(tree_path)
    leaves = list()
    for tip in tree.tips():
        if 'GC' in tip.name:
            tip.name = tip.name[3:].replace(' ', '_')
            leaves.append(tip.name)
    tree = tree.shear(leaves)
    tree.prune()
    for node in tree.non_tips():
        node.name = node.name.replace(' ', '_')
    return tree

def read_metadata(csv_path, tree_path=None):
    """
    Read GTDB metadata file and return taxonomy table
    """
    taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    def func(row):
        dat = dict(zip(taxlevels, row['gtdb_taxonomy'].split(';')))
        dat['species'] = dat['species'].split(' ')[1]
        dat['accession'] = row['accession'][3:]
        return pd.Series(data=dat)

    taxdf = pd.read_csv(csv_path, header=0, sep='\t')[['accession', 'gtdb_taxonomy']]\
                        .apply(func, axis=1)\
                        .set_index('accession')
    mask = np.array(acc[0] == 'G' for acc in taxdf.index)
    taxdf = taxdf.loc[mask]

    if tree_path is not None:
        # read tree, and then remove rows that are not in the tree
        tree = read_tree(tree_path)
        leaves = [tip.name for tip in tree.tips()]
        taxdf = taxdf.filter(leaves, axis=0)
        return taxdf, tree
    else:
        return taxdf
