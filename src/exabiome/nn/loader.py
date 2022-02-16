import argparse
import sys
from time import time
import warnings

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from hdmf.common import get_hdf5io

from ..sequence import AbstractChunkedDIFile, WindowChunkedDIFile, LazyWindowChunkedDIFile, RevCompFilter, DeepIndexFile, chunk_sequence, lazy_chunk_sequence, DIFileFilter
from ..utils import parse_seed, distsplit

import psutil
import os

def print_mem(msg=None):
    pid = os.getpid()
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    mem.rss/(1024**3)
    if msg is not None:
        msg = f'{msg} - '
    else:
        msg = ''
    print(f'{msg}{pid} - {mem.rss/1024**3:10.2f} - {mem.vms/1024**3:10.2f} - {mem.shared/1024**3:10.2f}', file=sys.stderr)



def check_window(window, step):
    if window is None:
        return None, None
    else:
        if step is None:
            step = window
        return window, step


def add_dataset_arguments(parser):
    group = parser.add_argument_group("Dataset options")

    group.add_argument('-W', '--window', type=int, default=None, help='the window size to use to chunk sequences')
    group.add_argument('-S', '--step', type=int, default=None, help='the step between windows. default is to use window size (i.e. non-overlapping chunks)')
    group.add_argument('--fwd_only', default=False, action='store_true', help='use forward strand of sequences only')
    group.add_argument('--tnf', default=False, action='store_true', help='use tetranucleotide frequencies as inputs')
    type_group = group.add_mutually_exclusive_group()
    type_group.add_argument('-C', '--classify', action='store_true', help='run a classification problem', default=False)
    type_group.add_argument('-M', '--manifold', action='store_true', help='run a manifold learning problem', default=False)
    choices = list(DeepIndexFile.taxonomic_levels) + ['all']
    group.add_argument('-t', '--tgt_tax_lvl', choices=choices, metavar='LEVEL', default='species',
                       help='the taxonomic level to predict. choices are domain, phylum, class, order, family, genus, species, all')

    return None


def dataset_stats(argv=None):
    """Read a dataset and print the number of samples to stdout"""

    parser =  argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    add_dataset_arguments(parser)
    parser.add_argument('--rank', type=int, help='subset the sequences based on world size and rank', default=None)
    parser.add_argument('--size', type=int, help='subset the sequences based on world size and rank', default=None)
    test_group = parser.add_argument_group('Test reading')
    test_group.add_argument('-T', '--test_read', default=False, action='store_true', help='test reading an element')
    test_group.add_argument('--mem', default=False, action='store_true', help='print memory usage before and after loading')
    test_group.add_argument('-L', '--lightning', default=False, action='store_true', help='test reading with DeepIndexDataModule')
    test_group.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=0)
    test_group.add_argument('-y', '--pin_memory', action='store_true', default=False, help='pin memory when loading data')
    test_group.add_argument('-f', '--shuffle', action='store_true', default=False, help='shuffle batches when training')
    test_group.add_argument('-b', '--batch_size', type=int, help='the number of workers to load data with', default=1)
    test_group.add_argument('-N', '--num_batches', type=int, help='the number of batches to load when testing read', default=None)
    test_group.add_argument('-s', '--seed', type=parse_seed, default='', help='seed for an 80/10/10 split before reading an element')
    test_group.add_argument('-l', '--load', action='store_true', default=False, help='load data into memory before running training loop')
    test_group.add_argument('-m', '--output_map', nargs=2, type=str, help='print the outputs map from one taxonomic level to another', default=None)


    args = parser.parse_args(argv)


    tr_len = None
    va_len = None
    if args.lightning:
        args.downsample = False
        args.loader_kwargs = dict()
        if args.mem:
            print_mem('before DeepIndexDataModule')
        before = time()
        data_mod = DeepIndexDataModule(hparams=args, keep_open=True)
        after = time()
        if args.mem:
            print_mem('after DeepIndexDataModule ')

        dataset = data_mod.dataset
        dataset.set_subset(train=True)
        tr_len = len(dataset)
        dataset.set_subset(validate=True)
        va_len = len(dataset)
        dataset.set_subset()
    else:
        if (args.rank != None and args.size == None) or (args.rank != None and args.size == None):
            print("You must specify both --rank and --size", file=sys.stderr)
            exit(1)

        kwargs = dict(path=args.input, hparams=args, keep_open=True, lazy_chunk=True)
        if args.rank != None:
            kwargs['rank'] = args.rank
            kwargs['size'] = args.size
        before = time()
        dataset = LazySeqDataset(**kwargs)
        dataset.load(sequence=False)
        after = time()

    print(f'Took {after - before} seconds to open {args.input}')
    difile = dataset.difile
    orig_difile = difile
    while not isinstance(orig_difile, DeepIndexFile):
        orig_difile = orig_difile.difile

    n_taxa = len(orig_difile.taxa_table)
    n_seqs = len(orig_difile.seq_table)

    n_samples = len(dataset)

    if args.window:
        n_disc = difile.n_discarded
        wlen = args.window
        step = args.step
        print((f'Splitting {n_seqs} sequences (from {n_taxa} species) into {wlen} '
               f'bp windows every {step} bps produces {n_samples} samples '
               f'(after discarding {n_disc} samples).'))
        if tr_len is not None:
            print(f'There are {tr_len} training samples and {va_len} validation samples')
    else:
        print(f'Found {n_seqs} sequences across {n_taxa} species. {n_samples} total samples')


    if args.test_read:
        dataset.close()
        from tqdm import tqdm
        print("Attempting to read training data")
        if args.lightning:
            if args.mem:
                print_mem('before train_dataloader   ')
            tr = data_mod.train_dataloader()
            if args.mem:
                print_mem('after train_dataloader    ')
        else:
            dataset.set_subset(train=True)
            kwargs = {'collate_fn': get_collater(dataset), 'shuffle': True, 'batch_size': 1}
            tr = DataLoader(dataset, **kwargs)

        tot = len(tr)
        if args.num_batches != None:
            stop = args.num_batches - 1
            tot = args.num_batches
        else:
            stop = tot - 1
        for idx, i in tqdm(enumerate(tr), total=tot):
            if idx == stop:
                break

        print("Attempting to read validation data")
        if args.lightning:
            va = data_mod.val_dataloader()
        else:
            dataset.set_subset(validate=True)
            kwargs.pop('shuffle')
            va = DataLoader(dataset, **kwargs)

        tot = len(va)
        if args.num_batches != None:
            stop = args.num_batches - 1
            tot = args.num_batches
        else:
            stop = tot - 1
        for idx, i in tqdm(enumerate(va), total=tot):
            if idx == stop:
                break

        #print("Attempting to read testing data")
        #for i in tqdm(te):
        #    continue
    if args.output_map is not None:
        tl1, tl2 = args.output_map
        ret = difile.taxa_table.get_outputs_map(tl1, tl2)
        bad = False
        for id2, id1 in enumerate(ret):
            mask = difile.taxa_table[tl2].get(np.s_[:], index=True) == id2
            t2_vals = difile.taxa_table[tl1].get(mask, index=True)
            if not np.all(t2_vals == id1):
                t2 = difile.taxa_table[tl2].vocabulary[id2]
                t1 = difile.taxa_table[tl1].vocabulary[id1]
                print('ERROR -- not all {tl2} {t2} have {tl1} {t1}')
                bad = True
        if not bad:
            print(f'taxonomic hierarchy for {tl1} to {tl2} okay')


class SplitCollater:

    def __init__(self, padval, freq=1.0, factors=[2, 4, 8]):
        self.padval = padval
        self.freq = freq
        self.factors = factors

    def __call__(self, samples):
        maxlen = 0
        l_idx = -1
        if isinstance(samples, tuple):
            samples = [samples]
        for i, X, y, seq_id in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]
        X_ret = list()
        y_ret = list()
        idx_ret = list()
        size_ret = list()
        seq_id_ret = list()
        for i, X, y, seq_id in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif), value=self.padval)
            X_ret.append(X_)
            y_ret.append(y)
            size_ret.append(X.shape[l_idx])
            idx_ret.append(i)
            seq_id_ret.append(seq_id)
        X_ret = torch.stack(X_ret)
        y_ret = torch.stack(y_ret)
        size_ret = torch.tensor(size_ret)
        idx_ret = torch.tensor(idx_ret)
        seq_id_ret = torch.tensor(seq_id_ret)

        if self.freq == 1.0 or rs.rand() < self.freq:
            f = factors[rs.randint(len(factors))]
            y_ret = y_ret.repeat_interleave(f)
            seq_id_ret = seq_id_ret.repeat_interleave(f)
            idx_ret = idx_ret.repeat_interleave(f)
            X_ret = X_ret.reshape((X_ret.shape[0] * f, X_ret.shape[0] // f))

            q = lens // f
            r = lens // f
            n_bad_chunks = b.shape[-1]//f - q
            bad_chunk_pos = torch.where(n_bad_chunks > 0)[0]
            start_bad_chunks = bad_chunk_pos + q[bad_chunk_pos]

        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


class SeqCollater:

    def __init__(self, padval):
        self.padval = padval

    def __call__(self, samples):
        maxlen = 0
        l_idx = -1
        if isinstance(samples, tuple):
            samples = [samples]
        for i, X, y, seq_id in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]
        X_ret = list()
        y_ret = list()
        idx_ret = list()
        size_ret = list()
        seq_id_ret = list()
        for i, X, y, seq_id in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif), value=self.padval)
            X_ret.append(X_)
            y_ret.append(y)
            size_ret.append(X.shape[l_idx])
            idx_ret.append(int(i))
            seq_id_ret.append(int(seq_id))
        X_ret = torch.stack(X_ret)
        y_ret = torch.stack(y_ret)
        size_ret = torch.tensor(size_ret)
        idx_ret = torch.tensor(idx_ret)
        seq_id_ret = torch.tensor(seq_id_ret)
        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


class TrainingSeqCollater:

    def __init__(self, padval):
        self.padval = padval

    def __call__(self, samples):
        maxlen = 0
        l_idx = -1
        if isinstance(samples, tuple):
            samples = [samples]
        for i, X, y, seq_id in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]
        X_ret = list()
        y_ret = list()
        for i, X, y, seq_id in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif), value=self.padval)
            X_ret.append(X_)
            y_ret.append(y)
        X_ret = torch.stack(X_ret)
        y_ret = torch.stack(y_ret)
        return (X_ret, y_ret)


def _check_collater(padval, seq_collater):
    if seq_collater is not None:
        return seq_collater
    elif padval is not None:
        return SeqCollater(padval)
    else:
        raise ValueError("must specify padval or seq_collater")

class GraphCollater:

    def __init__(self, node_ids, padval=None, seq_collater=None):
        self.collater = _check_collater(padval, seq_collater)
        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long)

    def __call__(self, samples):
        """
        A function to collate samples and return a sub-distance matrix
        """
        idx_ret, X_ret, y_idx, size_ret, seq_id_ret = self.collater(samples)

        # Get distances
        y_ret = self.node_ids[y_idx]
        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


class DistanceCollater:

    def __init__(self, dmat, padval=None, seq_collater=None):
        self.collater = _check_collater(padval, seq_collater)
        if len(dmat.shape) == 1:
            from scipy.spatial.distance import squareform
            dmat = squareform(dmat)
        #self.dmat = torch.as_tensor(dmat, dtype=torch.float).pow(2)
        #self.dmat = torch.as_tensor(dmat, dtype=torch.float).sqrt()
        self.dmat = torch.as_tensor(dmat/dmat.max(), dtype=torch.float)

    def __call__(self, samples):
        """
        A function to collate samples and return a sub-distance matrix
        """
        idx_ret, X_ret, y_idx, size_ret, seq_id_ret = self.collater(samples)

        # Get distances
        y_ret = self.dmat[y_idx][:, y_idx]
        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


class TnfCollater:
    def __init__(self, vocab):
        self.bases = 4**torch.arange(4)
        rcmap = torch.tensor([3, 2, 1, 0])
        canonical = list()
        noncanonical = list()
        palindromes = list()
        seen = torch.zeros(256, dtype=bool)
        for i in range(256):
            if seen[i]:
                continue
            ar = torch.zeros(4, dtype=int)
            ar[3], r = divmod(i, 64)
            ar[2], r = divmod(r, 16)
            ar[1], ar[0] = divmod(r, 4)
            rc = rcmap[ar.flip(0)]
            rc_i = rc.matmul(self.bases)
            if i < rc_i:
                canonical.append(i)
                noncanonical.append(rc_i)
            elif rc_i < i:
                canonical.append(rc_i)
                noncanonical.append(i)
            else:
                palindromes.append(i)
            seen[i] = True
            seen[rc_i] = True
        self.canonical = torch.tensor(canonical)
        self.noncanonical = torch.tensor(noncanonical)
        self.palindromes = torch.tensor(palindromes)

        # calculate a map to convert DNA characters into 0-4 encoding
        self.cmap = torch.zeros(128, dtype=int) - 1
        count = 0
        self.padval = None
        for i, c in enumerate(vocab):
            if c == 'A':
                self.cmap[i] = 0            # A
                count += 1
            elif c == 'T':
                self.cmap[i] = 3            # T
                count += 1
            elif c == 'C':
                self.cmap[i] = 1            # C
                count += 1
            elif c == 'G':
                self.cmap[i] = 2            # G
                count += 1
            elif c == 'N':
                self.padval = i
                count += 1
            if count == 5:
                break
        if self.padval is None:
            raise ValueError("Could not find 'N' character in vocab -- this is needed to pad sequences")


    def __call__(self, samples):
        l_idx = -1
        if isinstance(samples, tuple):
            samples = [samples]

        maxlen = 0
        for i, X, y, seq_id in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]

        X_ret = list()
        y_ret = list()
        idx_ret = list()
        size_ret = list()
        seq_id_ret = list()
        for i, X, y, seq_id in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif), value=self.padval)
            X_ret.append(X_)
            y_ret.append(y)
            size_ret.append(X.shape[l_idx])
            idx_ret.append(i)
            seq_id_ret.append(seq_id)

        # calculate tetranucleotide frequency
        chunks = torch.stack(X_ret)

        ## 1. hash 4-mers
        __seq = self.cmap[chunks]
        i4mers = torch.stack([__seq[:, 0:-3], __seq[:, 1:-2], __seq[:, 2:-1], __seq[:, 3:]], axis=2)
        mask = torch.any(i4mers < 0, axis=2)
        h4mers = i4mers.matmul(self.bases)       # hashed 4-mers
        h4mers[mask] = 256    # use 257 to mark any 4-mers that had ambiguous nucleotides

        ## 2. count hashed 4-mers i.e. count integers from between 0-257 inclusive
        tnf = torch.zeros((32, 257), dtype=float)
        for i in range(tnf.shape[0]):
            counts = torch.bincount(h4mers[i], minlength=257)
            tnf[i] = counts/i4mers.shape[1]

        ## 3. merge canonical 4-mers
        canon_tnf = torch.zeros((32, 136))
        canon_tnf[:, :len(self.canonical)] = tnf[:, self.canonical] + tnf[:, self.noncanonical]
        canon_tnf[:, len(self.canonical):] = tnf[:, self.palindromes]

        X_ret = canon_tnf
        y_ret = torch.stack(y_ret)
        size_ret = torch.tensor(size_ret)
        idx_ret = torch.tensor(idx_ret)
        seq_id_ret = torch.tensor(seq_id_ret)

        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


import torch.distributed as dist

def train_test_validate_split(indices, stratify=None, random_state=None,
                              test_size=None, train_size=None, validation_size=None):
    """
    Return train test validation split for the given indices
    Args:
        indices   : indices for the dataset
        kwargs    : any additional arguments to pass into torch.DataLoader
    """

    random_state = check_random_state(random_state)

    before = time()
    print("Splitting out training data", file=sys.stderr)
    train_idx, tmp_idx = train_test_split(indices,
                                          train_size=train_size,
                                          stratify=stratify,
                                          random_state=random_state)
    after = time()
    print(f'Took {after - before} seconds to split out training data', file=sys.stderr)

    if stratify is not None:
        stratify = stratify[tmp_idx]

    before = time()
    print("Splitting validation and test data", file=sys.stderr)
    test_idx, val_idx = train_test_split(tmp_idx,
                                         train_size=test_size,
                                         stratify=stratify,
                                         random_state=random_state)
    after = time()
    print(f'Took {after - before} seconds to split validation and test data', file=sys.stderr)

    train_idx = indices[train_idx]
    test_idx = indices[test_idx]
    val_idx = indices[val_idx]

    return train_idx, test_idx, val_idx


class DatasetSubset(Dataset):

    def __init__(self, dataset, index):
        self.index = index
        self.dataset = dataset

    def __getitem__(self, i):
        return self.dataset[self.index[i]]

    def __len__(self):
        return len(self.index)

    # def __getattr__(self, attr):
    #     return getattr(self.dataset, attr)


def train_test_loaders(dataset, random_state=None, downsample=None, **kwargs):
    """
    Return DataLoaders for training, validation, and test datasets.

    Parameters
    ----------
    dataset: LazySeqDataset
        the path to the DeepIndex file

    random_state: int or RandomState, default=None
        The seed to use for randomly splitting data

    downsample: float, default=None
        Fraction to downsample data to before setting up DataLoaders

    kwargs:
        any additional arguments to pass into torch.DataLoader. Note
        *shuffle* will be ignored for when creating validation and test
        loaders i.e. no shuffling for validation and testing
    """
    random_state = check_random_state(random_state)
    n_samples = len(dataset)
    test_size = int(n_samples/10)
    validation_size = test_size
    train_size = n_samples - 2 * test_size

    print(f'Permuting {len(dataset)} integers', file=sys.stderr)
    before = time()
    indices = random_state.permutation(np.arange(len(dataset), dtype=np.uint32))
    after = time()
    print(f'Took {after - before} to permute indices', file=sys.stderr)

    train_idx = indices[:train_size]
    validate_idx = indices[train_size:train_size + validation_size]
    test_idx = indices[train_size + validation_size:]

    #rank = dist.get_rank()

    # from mpi4py import MPI
    # comm =  MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # print(f'My rank is {rank}', file=sys.stderr)

    # if rank == 0:
    #     dataset.open()  # open the dataset so we can get labels for each sample
    #     before = time()
    #     stratify = dataset.sample_labels
    #     after = time()
    #     print(f'Took {after - before} seconds to get sample labels for stratifying splits', file=sys.stderr)

    #     before = time()
    #     train_idx, test_idx, validate_idx = train_test_validate_split(np.arange(len(dataset)),
    #                                                                   stratify=stratify,
    #                                                                   random_state=random_state,
    #                                                                   train_size=train_size,
    #                                                                   validation_size=validation_size,
    #                                                                   test_size=test_size,)
    #     after = time()
    #     print(f'Took {after - before} seconds to compute splits', file=sys.stderr)
    #     dataset.close()   # now close it so dataset can be pickled when passing it to DataLoader workers
    # else:
    #     train_idx = np.zeros(train_size)
    #     validate_idx = np.zeros(validation_size)
    #     test_idx = np.zeros(test_size)

    # if rank == 0:
    #     print(f'Attempting to broadcast train_idx', file=sys.stderr)
    # comm.Bcast(train_idx, root=0)
    # if rank == 0:
    #     print(f'Attempting to broadcast validate_idx', file=sys.stderr)
    # comm.Bcast(validate_idx, root=0)
    # if rank == 0:
    #     print(f'Attempting to broadcast test_idx', file=sys.stderr)
    # comm.Bcast(test_idx, root=0)

    if dataset.tnf:
        collater = TnfCollater(dataset.vocab)
    elif dataset.manifold:
        collater = DistanceCollater(dataset.distances, seq_collater=collater)
    elif dataset.graph:
        collater = GraphCollater(dataset.node_ids, seq_collater=collater)
    else:
        collater = SeqCollater(dataset.padval)


    if kwargs.get('num_workers', None) not in (None, 0):
        if dataset.difile is not None:
            msg = (f'Requesting {kwargs["num_workers"]} workers for loading data -- '
                   'closing dataset to avoid pickling error. '
                   'To suppress this warning, Set keep_open=False when constructing LazySeqDataset '
                   'or call dataset.close before passing to train_test_loaders')
            warnings.warn(msg)
            dataset.close()

    train_dataset = DatasetSubset(dataset, train_idx)
    test_dataset = DatasetSubset(dataset, test_idx)
    validate_dataset = DatasetSubset(dataset, validate_idx)

    s = kwargs.pop('size', 1)
    r = kwargs.pop('rank', -1)
    if s > 1 and r >= 0:
        train_dataset = Subset(train_dataset, distsplit(len(train_dataset), s, r))
        test_dataset = Subset(test_dataset, distsplit(len(test_dataset), s, r))
        validate_dataset = Subset(validate_dataset, distsplit(len(validate_dataset), s, r))

    tr_dl = DataLoader(train_dataset, collate_fn=collater, **kwargs)
    kwargs.pop('shuffle', None)
    va_dl = DataLoader(test_dataset, collate_fn=collater, **kwargs)
    te_dl = DataLoader(validate_dataset, collate_fn=collater, **kwargs)

    return (tr_dl, va_dl, te_dl)


def get_collater(dataset, inference=False):
    if dataset.tnf:
        return TnfCollater(dataset.vocab)
    elif dataset.manifold:
        return DistanceCollater(dataset.distances, seq_collater=collater)
    elif dataset.graph:
        return GraphCollater(dataset.node_ids, seq_collater=collater)
    else:
        if inference:
            return SeqCollater(dataset.padval)
        else:
            return TrainingSeqCollater(dataset.padval)


def get_loader(dataset, inference=False, **kwargs):
    """
    Return a DataLoader that loads data from the given Dataset

    Args:
        dataset (Dataset): the dataset to return a DataLoader for
        inference  (bool): whether or not we are running inference
    """
    # Find the LazySeqDataset so we can pull necessary information from it
    orig_dataset = dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    elif not isinstance(dataset, LazySeqDataset):
        msg = ("Unrecognized type for 'dataset'. "
               "Expected torch.utils.data.Subset or exabione.nn.loader.LazySeqDataset. "
               "Got %s" % type(dataset).__name__)
        raise ValueError(msg)

    collater = get_collater(dataset, inference=inference)

    return DataLoader(orig_dataset, collate_fn=collater, **kwargs)


class WORSampler(Sampler):
    """Without Replacement Sampler"""

    def __init__(self, length, rng=None, rank=0, size=1):
        super().__init__(None)
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(rng)
        self.rng = rng
        dtype = np.uint32
        if length > (2**32 - 1):
            dtype = np.uint64

        self.rank = rank
        self.size = size

        # trim will clip extra samples (i.e. length % size) so that each
        # rank has the same number of samples.
        # Use this later if we decide we don't want to trim tail.
        trim = True
        if size > 1:
            self.indices = np.arange(rank, length, size, dtype=dtype)
            if trim:
                self.indices = self.indices[:length // size]
        else:
            self.indices = np.arange(length, dtype=dtype)
        self.curr_len = len(self.indices)

    def __iter__(self):
        self.curr_len = len(self.indices)
        return self

    def __len__(self):
        return len(self.indices)

    def __next__(self):
        if self.curr_len == 0:
            raise StopIteration
        idx = self.rng.integers(self.curr_len)
        ret = self.indices[idx]
        self.indices[self.curr_len - 1], self.indices[idx] = self.indices[idx], self.indices[self.curr_len - 1]
        self.curr_len -= 1
        return ret


class SubsetDataLoader(DataLoader):

    def __init__(self, dataset, train=False, validate=False, test=False, **kwargs):
        super().__init__(dataset, **kwargs)
        self.train = train
        self.validate = validate
        self.test = test


    def __iter__(self):
        self.dataset.set_subset(train=self.train, validate=self.validate, test=self.test)
        return super().__iter__()


class DeepIndexDataModule(pl.LightningDataModule):

    def __init__(self, hparams, inference=False, keep_open=False, seed=None, rank=0, size=1):
        super().__init__()

        kwargs = dict(batch_size=hparams.batch_size)

        if inference:
            hparams.manifold = False
            hparams.graph = False
            self.dataset = LazySeqDataset(hparams=hparams, keep_open=keep_open)
        else:
            self.dataset = LazySeqDataset(hparams=hparams, keep_open=keep_open)
            self.dataset.load(sequence=hparams.load)
            kwargs['pin_memory'] = hparams.pin_memory
            kwargs['sampler'] = None
            self.dataset.set_subset(train=True)
            train_len = len(self.dataset)
            self.dataset.set_subset()
            kwargs['sampler'] = WORSampler(train_len, rng=seed, rank=rank, size=size)

        self._parallel_load = hparams.num_workers != None and hparams.num_workers > 0

        kwargs.update(hparams.loader_kwargs)
        kwargs['num_workers'] = hparams.num_workers
        if self._parallel_load:
            kwargs['multiprocessing_context'] = 'spawn'
            kwargs['worker_init_fn'] = self.dataset.worker_init
            kwargs['persistent_workers'] = True

        kwargs['collate_fn'] = get_collater(self.dataset, inference=inference)
        kwargs['shuffle'] = False

        self._loader_kwargs = kwargs

    def _check_close(self, train=False, validate=False, test=False):
        if self._loader_kwargs.get('num_workers', None) not in (None, 0):
            self.dataset.close()

    def train_dataloader(self):
        kwargs = self._loader_kwargs.copy()
        if self._parallel_load:
            self.dataset.close()
        return SubsetDataLoader(self.dataset, train=True, **kwargs)

    def val_dataloader(self):
        kwargs = self._loader_kwargs.copy()
        if self._parallel_load:
            self.dataset.close()
        kwargs.pop('sampler', None)
        return SubsetDataLoader(self.dataset, validate=True, **kwargs)

    def test_dataloader(self):
        return None


class LazySeqDataset(Dataset):
    """
    A torch Dataset to handle reading samples read from a DeepIndex file when using
    multiprocessing workers for loading data.

    Underlying a DeepIndex file is an HDF5 file, from which data is lazily read.
    If using multiple workers for data loading (i.e. *num_workers* > 0),
    the underly h5py.File must be closed before spawning workers because h5py.File
    handles are not picklable. For this reason, this object will close the DeepIndex
    file after reading everything it needs to set up the Dataset. This behavior
    can be overriden by passing *keep_open = True*. Alternatively, *open* can be called
    before using this Dataset in a loader. If using workers to load data, you must
    pass *worker_init* to the *worker_init_fn* argument when instantiating a DataLoader.

    Parameters
    ----------

    path: str
        the path to the file to create a dataset from. Alternatively, the argument *input*
        can be used on the *hparams* object or as a keyword argument

    keep_open: bool, default=False
        Keep file open after opening

    hparams: Namespace, default=None
        a argparse.Namespace object with the parameters to use for setting up the
        dataset. Alternatively, these parameters can be passed in individually as
        keyword arguments

    kwargs: dict
        the parameters to use for setting up the dataset


    Valid hparams keys
    ------------------
    input: str, default=None
        An alternative argument for *path*

    window: int, default=None
        The size of the sequence window chunks

    step: int, default=None
        The distance between sequence window chunks. set this to equal *window* for
        non-overlapping chunks

    fwd_only: bool, default=False
        Use only the forward strand

    load: bool, default=False
        Load data from the underlying HDF5 file into memory

    classify: bool, default=True
        Prep data for a classification loss function. i.e. return taxonomic labels like phylum or genus

    manifold: bool, default=False
        Prep data for a manifold/distance loss function

    graph: bool, default=False
        Prep data for a graph learning problem

    tgt_tax_lvl: str, default='species'
        the target taxonomic level for. This can be 'phylum', 'class', 'order', 'family', 'genus',
        'species' or 'all. For *manifold=True* or *graph=True*, this must be 'species'. Setting to
        'all' will return labels for each taxonomic level.

    ohe: bool, default=False
        One-hot encode sequences. By default, return indices


    References
    ----------

    PyTorch DataLoaders
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader


    """
    def __init__(self, path=None, keep_open=False, hparams=None, lazy_chunk=True, rank=0, size=1, **kwargs):
        self.__lazy_chunk = lazy_chunk
        kwargs.setdefault('input', None)
        kwargs.setdefault('window', None)
        kwargs.setdefault('step', None)
        kwargs.setdefault('fwd_only', False)
        kwargs.setdefault('classify', True)
        kwargs.setdefault('tgt_tax_lvl', 'species')
        kwargs.setdefault('manifold', False)
        kwargs.setdefault('graph', False)
        kwargs.setdefault('load', False)
        kwargs.setdefault('ohe', False)
        kwargs.setdefault('tnf', False)
        kwargs.setdefault('weighted', None)

        self.comm = kwargs.pop('comm', None)

        if hparams is not None:
            if not isinstance(hparams, argparse.Namespace):
                raise ValueError('hparams must be a Namespace object')
            for k, v in vars(hparams).items():
                kwargs[k] = v

        self.val_frac = kwargs.pop('val_frac', 0.2)

        hparams = argparse.Namespace(**kwargs)

        self.path = path or hparams.input
        self.hparams = hparams
        self.load_data = hparams.load

        # memoize any chunking we've done so we only need to do it once
        self._chunkings = dict()

        self.window, self.step = check_window(hparams.window, hparams.step)
        self.revcomp = not hparams.fwd_only

        self._world_size = size
        self._global_rank = rank

        self._label_dtype = torch.int64

        self._train_subset = False
        self._validate_subset = False
        self._test_subset = False

        # open to get dataset length
        self.open()
        self.__len = len(self.difile)
        if self.__lazy_chunk:
            if isinstance(self.difile, DIFileFilter):
                counts = self.difile.lut.copy()
                counts[1:] = counts[1:] - counts[:-1]
                self.taxa_counts = np.bincount(self.orig_difile.labels, weights=counts).astype(int)
                self.taxa_labels = np.arange(len(self.taxa_counts))
            else:
                self.taxa_counts = np.bincount(self.orig_difile.labels).astype(int)
                self.taxa_labels = np.arange(len(self.taxa_counts))
        else:
            self.taxa_labels, self.taxa_counts = np.unique(self.difile.labels, return_counts=True)
        self.label_names = self.difile.get_label_classes()

        self.vocab = self.orig_difile.get_vocab()
        if len(self.vocab) > 18:
            self.protein = True
            idx = np.where(self.vocab == '-')[0]
            if len(idx) > 0:
                self.padval = idx[0]
            else:
                warnings.warn("Could not find null value for protein sequences. Looking for '-'. Padding with %s" % self.vocab[0])
                self.padval = 0
        else:
            self.protein = False
            idx = np.where(self.vocab == 'N')[0]
            if len(idx) > 0:
                self.padval = idx[0]
            else:
                warnings.warn("Could not find null value for DNA sequences. Looking for 'N'. Padding with %s" % self.vocab[0])
                self.padval = 0
        self.vocab_len = len(self.vocab)

        self.manifold = False
        self.graph = False
        self.tnf = hparams.tnf
        self.distances = None
        self.node_ids = None

        if hparams.manifold:
            self.manifold = True
            self.distances = self.difile.distances.data[:]
            if hparams.tgt_tax_lvl != 'species':
                raise ValueError("must run manifold learning (-M) method with 'species' taxonomic level (-t)")
        elif hparams.graph:
            self.graph = True
            if hparams.tgt_tax_lvl != 'species':
                raise ValueError("must run graph learning (-M) method with 'species' taxonomic level (-t)")

            # compute the reverse look up to go from taxon id to node id
            leaves = self.difile.tree_graph.leaves[:]
            node_ids = np.zeros(leaves.max()+1)
            for i in range(len(leaves)):
                tid = leaves[i]
                if i < 0:
                    continue
                node_ids[tid] = i
            self.node_ids = node_ids
        elif hparams.classify:
            self.classify = True
            if hparams.weighted == 'phy':
                self.distances = self.difile.distances.data[:]
        else:
            self.classify = True
            if hparams.weighted == 'phy':
                self.distances = self.difile.distances.data[:]

        self.__ohe = hparams.ohe

        if not keep_open:
            self.close()

    def close(self):
        if self.io is not None:
            self.io.close()
        self.difile = None
        self.orig_difile = None
        self.io = None

    def get_graph(self):
        """Return a csr_matrix representation of the tree graph"""
        return self.difile.tree_graph.to_spmat()

    def open(self):
        """Open the HDMF file and set up chunks and taxonomy label"""
        if self.comm is not None:
            self.io = get_hdf5io(self.path, 'r', comm=self.comm, driver='mpio')
        else:
            self.io = get_hdf5io(self.path, 'r')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.orig_difile = self.io.read()

        if self._world_size > 1:
            self.orig_difile.set_sequence_subset(distsplit(len(self.orig_difile), self._world_size, self._global_rank))

        self.difile = self.orig_difile

        self.load(sequence=self.load_data)

        self.difile.set_label_key(self.hparams.tgt_tax_lvl)

        if self.window is not None:
            self.set_chunks(self.window, self.step)

        if self.revcomp:
            self.set_revcomp()

        self._set_subset(train=self._train_subset, validate=self._validate_subset, test=self._test_subset)

    @property
    def rank(self):
        return self._global_rank

    def worker_init(self, worker_id):
        # September 15, 2021, ajtritt
        # This print statement is necessary to avoid processings from hanging when they are started
        # after a Summit maintenance, processes would hang. I was able to track it down to line 62
        # of multiprocessing/popen_spawn_posix.py. I still do not know the real cause of the problem
        # but it it appears that writing to standard error after starting a multiprocessing.Process
        # keeps thing moving along.
        self.open()

    def set_chunks(self, window, step=None):
        if self.__lazy_chunk:
            self.difile = LazyWindowChunkedDIFile(self.difile, window, step)
        else:
            chunks = self._chunkings.get((window, step))
            if chunks is None:
                self._chunkings[(window, step)] = chunk_sequence(self.difile, window, step)
            self.difile = AbstractChunkedDIFile(self.difile, *self._chunkings[(window, step)])

    def set_revcomp(self, revcomp=True):
        self.difile = RevCompFilter(self.difile)

    def set_subset(self, train=False, validate=False, test=False):
        self._train_subset = train
        self._validate_subset = validate
        self._test_subset = test
        if self.difile is not None:
            self._set_subset(train=self._train_subset, validate=self._validate_subset, test=self._test_subset)

    def _set_subset(self, train=False, validate=False, test=False):
        if all((not train, not validate, not test)):
            self.difile.set_subset(None, None)
        else:
            counts = self.difile.get_counts(orig=True)
            val_counts = np.round(self.val_frac * counts).astype(int)
            train_counts = counts - val_counts
            if validate:
                self.difile.set_subset(val_counts, self.hparams.seed, starts=train_counts)
            elif test:
                raise ValueError("Cannot do this yet, and I may never do it, since we use held-out genomes for testing")
            else:
                self.difile.set_subset(train_counts, self.hparams.seed)
        self.__len = len(self.difile)

    def set_ohe(self, ohe=True):
        self.__ohe = ohe

    def __len__(self):
        return len(self.difile) if self.difile is not None else self.__len

    @staticmethod
    def _to_numpy(data):
        return data[:]

    @staticmethod
    def _to_torch(device=None, dtype=None):
        def func(data):
            return torch.tensor(data, device=device, dtype=dtype)
        return func

    def _check_load(self, data, transforms):
        if not isinstance(data.data, torch.Tensor):
            if not isinstance(transforms, (tuple, list)):
                transforms = [transforms]
            for tfm in transforms:
                data.transform(tfm)

    def load(self, sequence=False, device=None):
        _load = lambda x: x[:]
        self.orig_difile.seq_table['id'].transform(_load)
        self.orig_difile.seq_table['length'].transform(lambda x: x[:].astype(int))
        self.orig_difile.seq_table['sequence_index'].transform(_load)
        if sequence:
            self.orig_difile.seq_table['sequence_index'].target.transform(_load)


    def __getitem__(self, i):
        # get sequence
        try:
            item = self.difile[i]
        except ValueError as e:
            print(self._train_subset, self._validate_subset, self._test_subset, file=sys.stderr)
            raise e
        idx = item['id']
        seq = item['seq']
        label = item['label']
        seq_id = item.get('seq_idx', -1)
        ## one-hot encode sequence
        seq = torch.as_tensor(seq, dtype=torch.int64)
        if self.__ohe:
            seq = F.one_hot(seq, num_classes=self.vocab_len).float()
        seq = seq.T
        label = torch.as_tensor(label, dtype=self._label_dtype)
        return (idx, seq, label, seq_id)
