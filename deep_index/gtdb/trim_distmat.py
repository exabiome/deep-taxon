import argparse
from itertools import combinations
import h5py
import numpy as np

DISTANCES = 'distances'
LEAF_NAMES = 'leaf_names'

parser = argparse.ArgumentParser()
parser.add_argument('dist_h5', type=str, help='the HDF5 file with distance data (output by tree2dmat)')
parser.add_argument('output_base', type=str, help='the basename for output files')
args = parser.parse_args()

with h5py.File(args.dist_h5, 'r') as f:
    dist = f[DISTANCES][:]
    taxa_names = f[LEAF_NAMES][:].astype('U')

n_taxa = taxa_names.shape[0]
assert n_taxa*(n_taxa-1)//2 == dist.shape[0]
taxa_mask = np.ones(n_taxa, dtype=bool)

good_taxa = list()
bad_taxa = list()
# example: GCF_000296615.1
with open('%s.ncbi_paths.txt' % args.output_base, 'w') as f:
    for tax_idx, t in enumerate(taxa_names):
        if '_' not in t:
            bad_taxa.append(t)
            taxa_mask[tax_idx] = False
            continue
        good_taxa.append(t)
        source, prefix, number = t.split('_')
        accession = "%s_%s" % (prefix, number)
        dirs = ['genomes', 'all', prefix]
        for i in range(0, 9, 3):
            dirs.append(number[i:i+3])
        dirs.append("%s*" % accession)
        f.write("ftp.ncbi.nlm.nih.gov/%s/%s*genomic.fna.gz\n" % ("/".join(dirs), accession))
        f.write("ftp.ncbi.nlm.nih.gov/%s/%s*protein.faa.gz\n" % ("/".join(dirs), accession))

with open('%s.ignored_taxa.txt' % args.output_base, 'w') as f:
    for t in bad_taxa:
        print(t, file=f)

# remove all distances corresponding to bad taxa from condensed distance matrix
dist_mask = np.ones(dist.shape[0], dtype=bool)
for i in np.where(np.logical_not(taxa_mask))[0]:
    shift = i*n_taxa - (i+1)*(i+2)//2

    for j in range(i+1, n_taxa):
        dist_mask[shift + j] = False

    for j in range(0, i):
        dist_mask[j*n_taxa - (j+1)*(j+2)//2 + i] = False

dist = dist[dist_mask]
taxa_names = taxa_names[taxa_mask]

with h5py.File('%s.distances.filtered.h5' % args.output_base, 'w') as f:
    f.attrs['description'] = 'bad taxa (i.e. taxa from GTDB that are not in NCBI) were removed'
    f.create_dataset(DISTANCES, data=dist)
    dset = f.create_dataset(LEAF_NAMES, shape=taxa_names.shape, dtype=h5py.special_dtype(vlen=str))
    dset[:] = taxa_names
