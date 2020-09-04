import sys
import subprocess
import os.path

from ..utils import parse_logger


def get_ftp_path(accession, sequence_only=True):
    """
    Example accession from GTDB: RS_GCF_000978935.1

    """
    if accession.startswith('RS_') or accession.startswith('GB_'):
        accession = accession[3:]

    path = ['ftp.ncbi.nlm.nih.gov/genomes/all',
            accession[:3],
            accession[4:7],
            accession[7:10],
            accession[10:13],
            f'{accession}*']
    if sequence_only:
        path.append('*f[a,n]a.gz')

    return os.path.join(*path)


def ncbi_path(args):
    '''Print path at NCBI FTP site to stdout'''
    import argparse

    desc = 'Print path at NCBI FTP site to stdout '
    parser  = argparse.ArgumentParser(description=desc)
    parser.add_argument('accession', type=str, help='the accession of the genome to retreive')
    parser.add_argument('-f', '--file', action='store_true', default=False, help='accession is a file with a list of accessions, one per line')

    args = parser.parse_args(args)

    accessions = get_accessions(args)
    for acc in accessions:
        print(get_ftp_path(acc))


class Rsync:

    def __init__(self, dest, quiet=False):
        self.dest = dest
        self.logger = parse_logger(os.path.join(dest, 'rsync.log'))
        if quiet:
            import logging
            self.logger.setLevel(logging.WARNING)


    def __call__(self, accession):
        self.logger.info(f'fetching {accession}')

        source = get_ftp_path(accession)

        # -L --> --copy-links
        # -r --> --recursive
        # -R --> --relative
        # -t --> --times
        # -v --> --verbose
        cmd = f'rsync -LrRtv rsync://{source} {self.dest}'

        retcode = subprocess.call(cmd, shell=True)
        if retcode != 0:
            self.logger.warning(f'failed to rsync {accession}')
        else:
            self.logger.warning(f'successfully rsynced {accession}')

def get_accessions(args):
    if args.accession  == '-':
        accessions = [l[:-1] for l in sys.stdin]
    else:
        if args.file:
            with open(args.accession, 'r') as f:
                accessions = [l[:-1] for l in f]
        else:
            accessions = [args.accession]
    return accessions



def ncbi_fetch(args):
    '''Retrieve sequence data from NCBI FTP site using rsync'''
    import argparse

    desc = 'Retrieve sequence data from NCBI FTP site using rsync'
    epi = 'rsync will log to dest/rsync.log'
    parser  = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('accession', type=str, help='the accession of the genome to retreive')
    parser.add_argument('dest', type=str, help='the destination directory save the downloaded files to')
    parser.add_argument('-f', '--file', action='store_true', default=False, help='accession is a file with a list of accessions, one per line')
    parser.add_argument('-p', '--processes', type=int, help='the number of rsync subprocesses to run', default=1)
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='print less information to log file')

    args = parser.parse_args(args)

    accessions = get_accessions(args)

    if os.path.exists(args.dest):
        if not os.path.isdir(args.dest):
            raise ValueError(f'{args.dest} already exists as file')
    else:
        os.mkdir(args.dest)

    rsync = Rsync(args.dest, quiet=args.quiet)

    if args.processes > 1:
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(args.processes)
        pool.map(rsync, accessions)
    else:
        for acc in accessions:
            rsync(acc)
