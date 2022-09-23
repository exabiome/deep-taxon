import os
import deep_taxon.sequence.dna_table
from hdmf.common import get_hdf5io

def get_taxa_id(path):
    c, n = os.path.basename(path).split('_')[0:2]
    return c + '_' + n


def make_fof(argv=None):
    '''Find files and print paths for accessions'''
    import argparse
    import logging
    import sys
    import glob
    import h5py

    from skbio import TreeNode

    from ..utils import get_faa_path, get_fna_path, get_genomic_path


    epi = "fadir should be organized with the same structure as the NCBI FTP site"
    parser = argparse.ArgumentParser(epilog=epi)
    parser.add_argument('fadir', type=str, help='directory with NCBI sequence files')
    parser.add_argument('accessions', type=str, help='A file containing accessions')
    parser.add_argument('-t', '--tree', action='store_true', default=False, help='accessions are from a tree in Newick format')
    parser.add_argument('-f', '--hdmf_file', action='store_true', default=False, help='get accessions from a DeepIndex HDMF file')
    parser.add_argument('-A', '--append', action='store_true', default=False, help='include accessions')
    parser.add_argument('-T', '--taxonomy', action='store_true', default=False, help='include taxonomy')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-P', '--protein', action='store_true', default=False, help='get paths for protein files')
    grp.add_argument('-C', '--cds', action='store_true', default=False, help='get paths for CDS files')
    grp.add_argument('-G', '--genomic', action='store_true', default=False, help='get paths for genomic files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='log to stderr')


    args = parser.parse_args(args=argv)

    if not any([args.protein, args.cds, args.genomic]):
        args.genomic = True

    loglvl = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(stream=sys.stderr, level=loglvl, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    if args.tree:
        #############################
        # read tree
        #############################
        logger.info('reading tree from %s' % args.accessions)
        tree = TreeNode.read(args.accessions, format='newick')

        logger.info('getting taxa IDs from tree')
        tid_set = set()
        for tip in tree.tips():
            name = tip.name
            if name[0:2] in ('RS', 'GB'):
                name = name[3:]
            name = name.replace(' ', '_')
            tid_set.add(name)


        logger.info('getting Fasta paths from %s' % args.fadir)
        suffix = 'protein' if args.protein else 'genomic'
        pattern = "%s/all/G*/[0-9][0-9][0-9]/[0-9][0-9][0-9]/[0-9][0-9][0-9]/*/*%s*.gz" % (args.fadir, suffix)
        for path in glob.iglob(pattern):
            if 'cds' in path and not args.cds:
                continue
            if 'rna' in path:
                continue
            taxa_id = get_taxa_id(path)
            if taxa_id in tid_set:
                print(path, file=sys.stdout)
                tid_set.remove(taxa_id)

        for tid in sorted(tid_set):
            print(tid, file=sys.stderr)

    else:
        accessions = None
        taxonomy = False
        if args.hdmf_file:
            with get_hdf5io(args.accessions, 'r') as io: #h5py.File(args.accessions, 'r') as f:
                difile = io.read()
                for c in difile.taxa_table.columns:
                    c.transform(lambda x: x[:])
                for c in difile.genome_table.columns:
                    c.transform(lambda x: x[:])
                tdf = difile.taxa_table.to_dataframe()
                gdf = difile.genome_table.to_dataframe(index=True)
                tdf.iloc[gdf['rep_idx']]
                accessions = gdf['taxon_id']

        else:
            if args.taxonomy:
                print("-T is only valid with -f", file=sys.stderr)
                exit()
            if os.path.exists(args.accessions):
                with open(args.accessions, 'r') as f:
                    accessions = f.readlines()
            else:
                accessions = [args.accessions]

        func = get_genomic_path
        if args.cds:
            func = get_fna_path
        elif args.protein:
            func = get_faa_path
        for idx, line in enumerate(accessions.astype('U')):
            path = func(line.strip(), args.fadir)
            if args.taxonomy:
                tax = "\t".join(tdf.iloc[idx])
                path = f'{tax}\t{path}'
            elif args.append:
                path = f'{line}\t{path}'
            print(path, file=sys.stdout)


if __name__ == '__main__':
    make_fof()
