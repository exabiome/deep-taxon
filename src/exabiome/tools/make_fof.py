import os

def get_taxa_id(path):
    c, n = os.path.basename(path).split('_')[0:2]
    return c + '_' + n

if __name__ == '__main__':
    import argparse
    import logging
    import sys
    import glob

    from skbio import TreeNode

    from ..utils import get_faa_path, get_fna_path


    epi = "fadir should be organized with the same structure as the NCBI FTP site"
    parser = argparse.ArgumentParser(epilog=epi)
    parser.add_argument('fadir', type=str, help='directory with NCBI sequence files')
    parser.add_argument('accessions', type=str, help='A file containing accessions')
    parser.add_argument('-t', '--tree', action='store_true', default=False, help='accessions are from a tree in Newick format')
    parser.add_argument('-p', '--protein', action='store_true', default=False, help='get paths for protein files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='log to stderr')


    args = parser.parse_args()

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
        suffix = 'protein' if args.protein else 'cds_from_genomic'
        pattern = "%s/all/G*/[0-9][0-9][0-9]/[0-9][0-9][0-9]/[0-9][0-9][0-9]/*/*%s*.gz" % (args.fadir, suffix)
        for path in glob.iglob(pattern):
            taxa_id = get_taxa_id(path)
            if taxa_id in tid_set:
                print(path, file=sys.stdout)
                tid_set.remove(taxa_id)

        for tid in sorted(tid_set):
            print(tid, file=sys.stderr)

    else:
        func = get_faa_path if args.protein else get_fna_path
        with open(args.accessions, 'r') as f:
            for line in f.readlines():
                print(func(line.strip(), args.fadir), file=sys.stdout)
