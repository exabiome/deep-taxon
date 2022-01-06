from datetime import datetime
import argparse
from contextlib import contextmanager
import glob
import os
import sys
import logging
import warnings

import numpy as np


@contextmanager
def ccm(cond, cm):
    """
    A conditional context manager. This is useful for
    writing collective IO code that can run in serial

    e.g.
        with ccm(COMM.Get_size() > 1, dset.collective):
            ...
    """
    if cond:
        with cm:
            yield
    else:
        yield


def check_argv(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, str):
        argv = argv.strip().split()
    return argv


def parse_logger(string):
    if not string:
        ret = logging.getLogger('stdout')
        hdlr = logging.StreamHandler(sys.stdout)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)
    ret.setLevel(logging.INFO)
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return ret


def get_logger():
    return parse_logger('')


def check_directory(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError(f'{path} already exists as a file')
    else:
        os.makedirs(path)

def get_seed():
    return int(datetime.now().timestamp()*1000000) % (2**32 -1)


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


def get_genomic_path(acc, directory):
    if acc[:3] in ('RS_', 'GB_'):
        acc = acc[3:]
    l = [directory, 'all', acc[:3], acc[4:7], acc[7:10], acc[10:13], "%s*"%acc, "%s*_genomic.fna.gz" % acc]
    glob_str = os.path.join(*l)
    result = [s for s in glob.glob(glob_str) if not 'cds_from' in s and 'rna_from' not in s]
    if len(result) > 1:
        warnings.warn(f'more than one file matching {glob_str}, using most recently modified file')
        tmp = [(os.path.getmtime(f), f) for f in result]
        tmp.sort(key=lambda x: x[0])
        result = [tmp[-1][1]]
    if len(result) == 0:
        raise ValueError(f'no file matching {glob_str}')
    return result[0]


def get_accession(path):
    """
    Return the genome accession from a given path

    Example:
    given GCA_000309865.1_MMad_1.0/GCA_000309865.1_MMad_1.0_cds_from_genomic.fna.gz,
    return  GCA_000309865.1
    """
    basename = os.path.basename(path)
    return basename[:15]


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
    return _num_list(string, float)


def distsplit(dset_len, size, rank):
    q, r = divmod(dset_len, size)
    if rank < r:
        q += 1
        b = rank*q
        return np.arange(b, b+q)
    else:
        offset = (q+1)*r
        b = (rank - r)*q + offset
        return np.arange(b, b+q)
