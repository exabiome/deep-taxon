from datetime import datetime
import argparse


def parse_seed(string):
    if string:
        try:
            i = int(string)
            if i > 2**32 - 1:
                raise ValueError(string)
            return i
        except :
            raise argparse.ArgumentTypeError(f'{string} is not a valid seed')
    else:
        return int(datetime.now().timestamp())


def _get_path_helper(acc, directory, sfx):
    if acc[:3] in ('RS_', 'GB_'):
        acc = acc[3:]
    l = [directory, 'all', acc[:3], acc[4:7], acc[7:10], acc[10:13], "%s*"%acc, "%s*%s" % (acc, sfx)]
    glob_str = os.path.join(*l)
    return glob_str


def get_faa_path(acc, directory):
    return _get_path_helper(acc, directory, '_protein.faa.gz')


def get_fna_path(acc, directory):
    return _get_path_helper(acc, directory, '_cds_from_genomic.fna.gz')
