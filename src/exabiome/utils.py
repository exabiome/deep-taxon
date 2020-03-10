from datetime import datetime
import argparse
import glob
import os


def get_seed():
    return int(datetime.now().timestamp())


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
        return get_seed()


def _get_path_helper(acc, directory, sfx):
    if acc[:3] in ('RS_', 'GB_'):
        acc = acc[3:]
    l = [directory, 'all', acc[:3], acc[4:7], acc[7:10], acc[10:13], "%s*"%acc, "%s*%s" % (acc, sfx)]
    glob_str = os.path.join(*l)
    result = glob.glob(glob_str)
    if len(result) > 1:
        raise ValueError(f'more than one file matching {glob_str}')
    if len(result) == 0:
        raise ValueError(f'no file matching {glob_str}')
    return result[0]


def get_faa_path(acc, directory):
    return _get_path_helper(acc, directory, '_protein.faa.gz')


def get_fna_path(acc, directory):
    return _get_path_helper(acc, directory, '_cds_from_genomic.fna.gz')


def _num_list(string, t):
    if isinstance(string, (list, tuple)):
        return string
    elif isinstance(string, str):
        if ':' in string:
            ar = [t(a) for a in string.split(":")]
            return list(range(ar[0], ar[1]+1))
        else:
            return list(map(t, string.split(",")))
    else:
        raise argparse.ArgumentTypeError(f'cannot parse {string} as list of {t}')


def int_list(string):
    return _num_list(string, int)


def float_list(string):
    return _num_list(string, int)
