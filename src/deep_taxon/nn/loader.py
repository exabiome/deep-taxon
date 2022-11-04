import argparse
import sys
from time import time
import warnings

import psutil
import os

from hdmf.common import get_hdf5io
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from .collaters import get_collater
from .samplers import DSSampler, WORSampler
from .umap import NeighborGraphSampler
from ..sequence import LazyWindowChunkedDIFile, DeepIndexFile, chunk_sequence, lazy_chunk_sequence
from ..utils import parse_seed, distsplit, balsplit, get_logger, log


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
    parser.add_argument('--mpi', default=False, action='store_true', help='user MPI rank/size')
    parser.add_argument('-P', '--n_partitions', type=int, help='the number of partitions to use', default=1)
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
    if (args.rank != None and args.size == None) or (args.rank != None and args.size == None):
        print("You must specify both --rank and --size", file=sys.stderr)
        exit(1)
    elif args.rank is None and args.size is None:
        args.rank = 0
        args.size = 1

    comm = None
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        print("MPI imported.", file=sys.stderr)
        if args.mpi:
            args.rank = comm.Get_rank()
            args.size = comm.Get_size()
            print(f"Reading for rank {args.rank} of {args.size}", file=sys.stderr)
        comm = None
    except:
        pass

    io = get_hdf5io(args.input, 'r')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        difile = io.read()
    difile.set_label_key(args.tgt_tax_lvl)

    if args.lightning:
        args.downsample = False
        args.loader_kwargs = dict()
        if args.mem:
            print_mem('before DeepIndexDataModule')
        before = time()
        data_mod = DeepIndexDataModule(difile=difile, hparams=args, keep_open=True, rank=args.rank, size=args.size, comm=comm)
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
        kwargs = dict(path=args.input, hparams=args, keep_open=True, lazy_chunk=True, difile=difile)
        if args.rank != None:
            kwargs['rank'] = args.rank
            kwargs['size'] = args.size
        if comm is not None:
            kwargs['comm'] = comm
        before = time()
        dataset = LazySeqDataset(load=False, **kwargs)
        after = time()

    io.close()

    print(f'Took {after - before} seconds to open {args.input}')
    difile = dataset.difile
    n_taxa = np.sum(np.bincount(difile.labels) > 0)
    n_seqs = difile.n_seqs

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

def read_all_seqs(fa_path):

    import skbio
    from skbio.sequence import DNA, Protein
    kwargs = {'format': 'fasta', 'constructor': DNA, 'validate': False}
    seqs = list()
    for seq in skbio.io.read(fa_path, **kwargs):
        seqs.append(seq.values.astype('U'))
    return "".join(np.concatenate(seqs))

def check_loaded_sequences(argv=None):
    """Read a dataset and print the number of samples to stdout"""
    from ..utils import get_genomic_path

    parser =  argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('fadir', type=str, help='directory with NCBI sequence files')
    add_dataset_arguments(parser)
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument('--train', action='store_true', help='use training data loader', default=False)
    type_group.add_argument('--validate', action='store_true', help='use validation data loader', default=False)
    parser.add_argument('--rank', type=int, help='subset the sequences based on world size and rank', default=0)
    parser.add_argument('--size', type=int, help='subset the sequences based on world size and rank', default=1)
    parser.add_argument('-P', '--n_partitions', type=int, help='the number of partitions to use', default=1)
    parser.add_argument('-b', '--batch_size', type=int, help='the number of workers to load data with', default=1)
    test_group = parser.add_argument_group('Test reading')
    test_group.add_argument('-N', '--num_batches', type=int, help='the number of batches to load when testing read', default=1)
    test_group.add_argument('-s', '--seed', type=parse_seed, default='', help='seed for an 80/10/10 split before reading an element')


    args = parser.parse_args(argv)
    logger = get_logger()

    tr_len = None
    va_len = None
    args.downsample = False
    args.loader_kwargs = dict()
    args.load = False
    args.pin_memory = False
    args.num_workers = 0
    args.shuffle = False
    logger.info(f"loading data from {args.input}")
    data_mod = DeepIndexDataModule(hparams=args, keep_open=True, rank=args.rank, size=args.size)
    dataset = data_mod.dataset

    if not (args.train or args.validate):
        args.train = True

    if args.train:
        logger.info(f"getting training data loader")
        tr = data_mod.train_dataloader()
    else:
        logger.info(f"getting validation data loader")
        tr = data_mod.val_dataloader()

    stop = args.num_batches - 1


    logger.info(f"Checking {args.num_batches} batches with a batch size of {args.batch_size}")
    correct = 0
    rc = 0
    n_samples = 0
    for idx, (seqs, labels) in tqdm(enumerate(tr), total=args.num_batches):
        print(seqs, labels)
        for i in range(len(seqs)):
            n_samples += 1
            func = get_genomic_path
            tid = dataset.difile.taxa_table.taxon_id.data[labels[i]]
            path = get_genomic_path(tid, args.fadir)
            seq = read_all_seqs(path)
            sample = "".join(dataset.vocab[seqs[i]]).rstrip('N')
            if sample in seq:
                correct += 1
            else:
                sample = "".join(dataset.vocab[dataset.difile.rcmap[seqs[i]]])[::-1].rstrip('N')
                if sample in seq:
                    correct += 1
                    rc += 1
        if idx == stop:
            break

    logger.info(f"{correct} ({rc} revcomped) samples of {n_samples} existed in the respective genomes returned by the loader")


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


class SubsetDataLoader(DataLoader):

    def __init__(self, dataset, train=False, validate=False, test=False, **kwargs):
        if 'sampler' not in kwargs:
            kwargs['sampler'] = DSSampler(dataset.get_subset_len(train=train, vaidate=validate, test=test))
        super().__init__(dataset, **kwargs)
        self.train = train
        self.validate = validate
        self.test = test

    def __len__(self):
        return (len(self.sampler) - 1) // self.batch_size + 1

    def __iter__(self):
        self.dataset.set_subset(train=self.train, validate=self.validate, test=self.test)
        return super().__iter__()


class fast_dataset(Dataset):
    def __init__(self, num_samples, sample_length):
        self.x = torch.zeros((num_samples, sample_length), device='cuda', dtype=torch.half)
        self.y = torch.zeros(num_samples, device='cuda', dtype=torch.float64)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return None

class new_collate_fxn():
    def __init__(self, batch_size, sample_length):
        self.x = torch.zeros((batch_size, sample_length), device='cuda', dtype=torch.half)
        self.y = torch.zeros(batch_size, device='cuda', dtype=torch.float64)
        #self.x = torch.zeros((batch_size, sample_length), dtype=torch.half).to('cuda')
        #self.y = torch.zeros(batch_size, dtype=torch.float64).to('cuda')
    def __call__(self, samples):
        return self.x, self.y


class FastDataModule(pl.LightningDataModule):
    def __init__(self, difile, hparams, inference=False, keep_open=False, seed=None, rank=0, size=1, **lsd_kwargs):
        super().__init__()
        kwargs = dict(batch_size=hparams.batch_size)
        self._loader_kwargs = kwargs
        self.batch_size = hparams.batch_size
        self.window_len = hparams.window
        self.num_samples = 1500000 #25,000,000 (at 16bit) takes up 42.92gb on a card
        self.val_pct = 0.1
        self.train_samples = int(self.num_samples * (1 - self.val_pct))
        self.valid_samples = int(self.num_samples * self.val_pct)

    def setup(self, stage=None):
        self.train_ds = fast_dataset(self.train_samples, self.window_len)
        self.valid_ds = fast_dataset(self.valid_samples, self.window_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                        collate_fn=new_collate_fxn(self.batch_size, self.window_len))

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size,
                        collate_fn=new_collate_fxn(self.batch_size, self.window_len))

    def test_dataloader(self):
        return None


class DeepIndexDataModule(pl.LightningDataModule):

    def __init__(self, difile, hparams, inference=False, keep_open=False, seed=None, rank=0, size=1, **lsd_kwargs):
        super().__init__()

        kwargs = dict(batch_size=hparams.batch_size)
        if seed is None:
            seed = parse_seed('')

        self.seed = seed
        self.rank = rank
        self.size = size
        self.n_partitions = hparams.n_partitions
        self.batch_size = hparams.batch_size
        self.n_batches = hparams.n_batches
        self.umap = hparams.umap
        self.sanity = hparams.sanity

        log("Creating LazySeqDataset", print_msg=rank==0)
        if inference:
            hparams.manifold = False
            hparams.graph = False
            self.dataset = LazySeqDataset(difile, hparams=hparams, keep_open=keep_open, **lsd_kwargs)
        else:
            self.dataset = LazySeqDataset(difile, hparams=hparams, keep_open=keep_open, rank=rank, size=size, **lsd_kwargs)
            # self.dataset.load(sequence=hparams.load)
            kwargs['pin_memory'] = hparams.pin_memory

            self._tr_sampler = None
            self._val_sampler = None

        self._parallel_load = hparams.num_workers != None and hparams.num_workers > 0

        kwargs.update(hparams.loader_kwargs)
        kwargs['num_workers'] = hparams.num_workers
        if self._parallel_load:
            kwargs['multiprocessing_context'] = 'spawn'
            #kwargs['worker_init_fn'] = self.dataset.worker_init
            kwargs['persistent_workers'] = True

        log("Getting collater", print_msg=rank==0)
        kwargs['collate_fn'] = get_collater(self.dataset, inference=inference, condensed=hparams.condensed)
        kwargs['shuffle'] = False

        self._loader_kwargs = kwargs

    def _check_close(self, train=False, validate=False, test=False):
        if self._loader_kwargs.get('num_workers', None) not in (None, 0):
            self.dataset.close()

    def get_min(self, val, batch_size):
        if self.size > 1:
            import torch.distributed as dist
            ret = torch.tensor(val).cuda()
            dist.all_reduce(ret, op=dist.ReduceOp.MIN)
            ret = int(ret.cpu())
            if (val - ret) < batch_size:
                ret = val
            return ret
        else:
            return val

    def train_dataloader(self):
        kwargs = self._loader_kwargs.copy()
        if self._tr_sampler is None:
            # set up the training sampler - shuffle for training
            train_len = self.get_min(self.dataset.get_subset_len(train=True), self.batch_size)
            s_kwargs = dict(rng=self.seed, n_partitions=self.n_partitions, part_smplr_rng=self.seed+self.rank)
            if self.sanity:
                s_kwargs['max_samples'] = self.batch_size * self.sanity
            if args.umap:
                self._tr_sampler = NeighborGraphSampler(self.dataset.graph,
                                                        self.dataset.difile,
                                                        self.dataset.difile.get_counts(),
                                                        n_batches=self.n_batches,
                                                        batch_size=self.batch_size,
                                                        edge_sampler=WORSampler)
            else:
                self._tr_sampler = WORSampler(train_len, **s_kwargs)
        kwargs['sampler'] = self._tr_sampler
        return SubsetDataLoader(self.dataset, train=True, **kwargs)

    def val_dataloader(self):
        kwargs = self._loader_kwargs.copy()
        if self._val_sampler is None:
            # set up the validation sampler - DO NOT shuffle for validation
            val_len = self.get_min(self.dataset.get_subset_len(validate=True), self.batch_size)
            s_kwargs = dict(n_partitions=self.n_partitions, part_smplr_rng=self.seed+self.rank)
            if self.sanity:
                s_kwargs['max_samples'] = self.batch_size * self.sanity // 4
            if self.umap:
                self._val_sampler = NeighborGraphSampler(self.dataset.graph,
                                                         self.dataset.difile,
                                                         self.dataset.difile.get_counts(),
                                                         n_batches=self.n_batches,
                                                         batch_size=self.batch_size,
                                                         edge_sampler=DSSampler)
            else:
                self._val_sampler = DSSampler(val_len, **s_kwargs)
        kwargs['sampler'] = self._val_sampler
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
    def __init__(self, difile=None, keep_open=False, hparams=None, lazy_chunk=True, rank=0, size=1, **kwargs):
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

        self.hparams = hparams

        self.window, self.step = check_window(hparams.window, hparams.step)
        self.revcomp = not hparams.fwd_only

        self._world_size = size
        self._global_rank = rank

        self._label_dtype = torch.int64

        self._train_subset = False
        self._validate_subset = False
        self._test_subset = False

        self._val_counts = None
        self._train_counts = None


        self.manifold = False
        self.graph = False
        self.neighbor_graph = None
        self.tnf = hparams.tnf
        self.__ohe = hparams.ohe

        distances = False
        tree_graph = False

        if hparams.manifold:
            self.manifold = True
            if hparams.tgt_tax_lvl != 'species':
                raise ValueError("must run manifold learning (-M) method with 'species' taxonomic level (-t)")
            distances = True
        elif hparams.graph:
            self.graph = True
            if hparams.tgt_tax_lvl != 'species':
                raise ValueError("must run graph learning (-M) method with 'species' taxonomic level (-t)")
            tree_graph = tree_graph
        elif hparams.classify:
            self.classify = True
            if hparams.weighted == 'phy':
                distances = True
        elif hparams.umap:
            self.classify = True
            self.neighbor_graph = get_neighbor_graph(difile.distances.data, n_neighbors=hparams.n_neighbors)
        else:
            self.classify = True
            if hparams.weighted == 'phy':
                distances = True

        # open to get dataset length

        self.difile = LazyWindowChunkedDIFile(difile, self.window, self.step,
                                              revcomp=self.revcomp,
                                              rank=self._global_rank, size=self._world_size,
                                              tree_graph=tree_graph,
                                              load=kwargs['load'],
                                              shmem=kwargs['shmem'])
        self._set_subset(train=self._train_subset, validate=self._validate_subset, test=self._test_subset)

        self.__len = len(self.difile)
        self._orig_len = self.__len
        self.__n_classes = self.difile.n_classes

        counts = self.difile.get_counts(orig=True)
        self._val_counts = np.round(self.val_frac * counts).astype(int)
        self._train_counts = counts - self._val_counts

        self.vocab = self.difile.vocab
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


    @property
    def n_classes(self):
        return self.__n_classes

    def get_graph(self):
        """Return a csr_matrix representation of the tree graph"""
        return self.difile.tree_graph

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

    def get_subset_len(self, train=False, validate=False, test=False):
        rc = 2 if self.revcomp else 1
        if train:
            return self._train_counts.sum() * rc
        elif validate:
            return self._val_counts.sum() * rc
        elif test:
            raise ValueError("We don't support a test subset")
        return self._orig_len

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
            if validate:
                self.difile.set_subset(self._val_counts, self.hparams.seed, starts=self._train_counts)
            elif test:
                raise ValueError("Cannot do this yet, and I may never do it, since we use held-out genomes for testing")
            else:
                self.difile.set_subset(self._train_counts, self.hparams.seed)
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
        #seq = torch.as_tensor(seq, dtype=torch.int64)
        seq = torch.as_tensor(seq)
        if self.__ohe:
            seq = F.one_hot(seq, num_classes=len(self.difile.vocab)).float()
        label = torch.as_tensor(label, dtype=self._label_dtype)
        return (idx, seq, label, seq_id)
