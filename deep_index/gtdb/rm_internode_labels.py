import dendropy
from dendropy import treecalc
import argparse
from itertools import combinations
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('tree', type=str, help='the tree file to convert')
parser.add_argument('out_tree', type=str, help='the output file')
args = parser.parse_args()

tree = dendropy.Tree.get(path=args.tree, schema='newick')
print(args.out_tree)

tree.write(path=args.out_tree, schema='newick', suppress_leaf_taxon_labels=False, suppress_leaf_node_labels=True, suppress_internal_taxon_labels=True, suppress_internal_node_labels=True)

