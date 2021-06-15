import argparse
import sys
import warnings

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from hdmf.common import get_hdf5io

from ..sequence import AbstractChunkedDIFile, WindowChunkedDIFile, RevCompFilter, DeepIndexFile, chunk_sequence
from ..utils import parse_seed


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
    type_group = group.add_mutually_exclusive_group()
    type_group.add_argument('-C', '--classify', action='store_true', help='run a classification problem', default=False)
    type_group.add_argument('-M', '--manifold', action='store_true', help='run a manifold learning problem', default=False)
    group.add_argument('-t', '--tgt_tax_lvl', choices=DeepIndexFile.taxonomic_levels, metavar='LEVEL', default='species',
                       help='the taxonomic level to predict. choices are phylum, class, order, family, genus, species')

    return None


def dataset_stats(argv=None):
    """Read a dataset and print the number of samples to stdout"""

    parser =  argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    add_dataset_arguments(parser)
    test_group = parser.add_argument_group('Test reading')
    test_group.add_argument('-T', '--test_read', default=False, action='store_true', help='test reading an element')
    test_group.add_argument('-s', '--seed', type=parse_seed, default=None, help='seed for an 80/10/10 split before reading an element')
    test_group.add_argument('-l', '--load', action='store_true', default=False, help='load data into memory before running training loop')
    test_group.add_argument('-m', '--output_map', nargs=2, type=str, help='print the outputs map from one taxonomic level to another', default=None)


    args = parser.parse_args(argv)
    dataset, io = process_dataset(args)
    difile = io.read()

    n_taxa = len(difile.taxa_table)
    n_seqs = len(difile.seq_table)

    n_samples = len(dataset)
    wlen = args.window
    step = args.step
    if wlen is not None:
        print(("Splitting %d sequences (from %d species) into %d "
               "bp windows every %d bps produces %d samples") % (n_seqs, n_taxa, wlen, step, n_samples))
    else:
        print(("Found %d sequences across %d species. %d total samples") % (n_seqs, n_taxa, n_samples))

    if args.test_read:
        print("Attempting to read training data")
        tr, va, te = train_test_loaders(dataset, random_state=args.seed, downsample=None, distances=args.manifold)
        from tqdm import tqdm
        for i in tqdm(tr):
            continue
        print("Attempting to read validation data")
        for i in tqdm(va):
            continue
        print("Attempting to read testing data")
        for i in tqdm(te):
            continue
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


def read_dataset(path):
    hdmfio = get_hdf5io(path, 'r')
    difile = hdmfio.read()
    dataset = SeqDataset(difile)
    return dataset, hdmfio


def process_dataset(args, path=None):
    """
    Process *input* argument and return dataset and HDMFIO object
    Args:
        args (Namespace):       command-line arguments passed by parser
    """
    dataset, io = read_dataset(path or args.input)

    if not hasattr(args, 'classify'):
        raise ValueError('Parser must check for classify/regression/manifold '
                         'to determine the number of outputs')
    if args.classify:
        dataset.set_classify(True)
        dataset.difile.set_label_key(args.tgt_tax_lvl)
        args.n_outputs = dataset.difile.n_outputs
    elif args.manifold:
        if args.tgt_tax_lvl != 'species':
            raise ValueError("must run manifold learning (-M) method with 'species' taxonomic level (-t)")
        dataset.set_classify(True)
    else:
        raise ValueError('classify (-C) or manifold (-M) should be set')

    args.window, args.step = check_window(args.window, args.step)

    # Process any arguments that impact how we set up the dataset
    dataset.set_ohe(False)
    if args.window is not None:
        dataset.set_chunks(args.window, args.step)
        #dataset.difile = WindowChunkedDIFile(dataset.difile, args.window, args.step)
    if not getattr(args, 'fwd_only', False):
        dataset.set_revcomp()
    if args.load:
        dataset.load()

    return dataset, io


class GraphCollater:
    def __init__(self, node_ids):
        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long)

    def __call__(self, samples):
        """
        A function to collate samples and return a sub-distance matrix
        """
        idx_ret, X_ret, y_idx, size_ret, seq_id_ret = collate(samples)

        # Get distances
        y_ret = self.node_ids[y_idx]
        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)

class DistanceCollater:
    def __init__(self, dmat):
        if len(dmat.shape) == 1:
            from scipy.spatial.distance import squareform
            dmat = squareform(dmat)
        self.dmat = torch.as_tensor(dmat, dtype=torch.float).pow(2)

    def __call__(self, samples):
        """
        A function to collate samples and return a sub-distance matrix
        """
        idx_ret, X_ret, y_idx, size_ret, seq_id_ret = collate(samples)

        # Get distances
        y_ret = self.dmat[y_idx][:, y_idx]
        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)

def collate(samples):
    """
    A function to collate variable length sequence samples
    """
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
            X_ = F.pad(X, (0, dif))
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
    return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


class SeqDataset(Dataset):
    """
    A torch Dataset to handle reading samples read from a DeepIndex file
    """

    def __init__(self, difile, classify=True):
        self.difile = difile
        self.set_classify(classify)
        self._target_key = 'class_label' if classify else 'embedding'
        self.vocab_len = len(self.difile.seq_table['sequence'].target.elements)
        self.__ohe = True

    def set_classify(self, classify):
        self._classify = classify
        if classify:
            self._label_key = 'id'
            self._label_dtype = torch.int64
        else:
            self._label_key = 'embedding'
            self._label_dtype = torch.float32

        # THIS HAS BEEN A MAJOR SOURCE OF PAIN. DYNAMIC TABLE NEEDS BETTER SLICING
        # It should be possible to select individual columns without haveing to modify
        # the state of the underlying DynamicTable
        self.difile.set_label_key(self._label_key)

    def set_chunks(self, window, step=None):
        self.difile = WindowChunkedDIFile(self.difile, window, step)

    def set_revcomp(self, revcomp=True):
        self.difile = RevCompFilter(self.difile)

    def set_ohe(self, ohe=True):
        self.__ohe = ohe

    def __len__(self):
        return len(self.difile)

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

    def load(self, device=None):
        def _load(data):
            return data[:]

        tfm = self._to_torch(device)
        def to_sint(data):
            return data[:].astype(np.int16)
        self._check_load(self.difile.seq_table['sequence'].target, [to_sint, tfm])
        self._check_load(self.difile.taxa_table[self._label_key], tfm)
        self._check_load(self.difile.distances, tfm)

        for col in self.difile.seq_table.children:
            if col.name == 'sequence':
                continue
            col.transform(_load)

        for col in self.difile.taxa_table.children:
            if col.name == self._label_key:
                continue
            col.transform(_load)


    def __getitem__(self, i):
        # get sequence
        item = self.difile[i]
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


def train_test_validate_split(data, stratify=None, random_state=None,
                              test_size=0.1, train_size=0.8, validation_size=0.1):
    """
    Return train test validation split of given data
    test_size, train_size, validation_size will all be normalized before subsequent
    calls to train_test_split
    Args:
        data (str): the path to the DeepIndex file
        kwargs    : any additional arguments to pass into torch.DataLoader
    """
    indices = np.arange(len(data))

    tot = train_size + test_size + validation_size
    train_size /= tot
    test_size /= tot
    validation_size /= tot

    random_state = check_random_state(random_state)

    train_idx, tmp_idx = train_test_split(indices,
                                          train_size=train_size,
                                          stratify=stratify,
                                          random_state=random_state)

    if stratify is not None:
        stratify = stratify[tmp_idx]

    tot = test_size + validation_size
    test_size /= tot
    validation_size /= tot

    test_idx, val_idx = train_test_split(tmp_idx,
                                         train_size=test_size,
                                         stratify=stratify,
                                         random_state=random_state)

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
    index = np.arange(len(dataset))
    #stratify = dataset.difile.labels
    stratify = dataset.sample_labels
    if downsample is not None:
        index, _, stratify, _ = train_test_split(index, stratify,
                                                 train_size=downsample,
                                                 random_state=random_state)

    train_idx, test_idx, validate_idx = train_test_validate_split(index,
                                                                  stratify=stratify,
                                                                  random_state=random_state)

    collater = collate
    if dataset.manifold:
        collater = DistanceCollater(dataset.distances)
    elif dataset.graph:
        collater = GraphCollater(dataset.node_ids)

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

    tr_dl = DataLoader(train_dataset, collate_fn=collater, **kwargs)
    kwargs.pop('shuffle', None)
    va_dl = DataLoader(test_dataset, collate_fn=collater, **kwargs)
    te_dl = DataLoader(validate_dataset, collate_fn=collater, **kwargs)

    return (tr_dl, va_dl, te_dl)

def get_loader(dataset, distances=False, **kwargs):
    """
    Return a DataLoader that loads data from the given Dataset

    Args:
        dataset (Dataset): the dataset to return a DataLoader for
        distances  (bool): whether or not to return distances for a batch
    """
    collater = collate
    if distances:
        if dataset.difile.distances is None:
            raise ValueError('DeepIndexFile {dataset.difile} does not contain distances')
        collater = DistanceCollater(dataset.difile.distances.data[:])
    return DataLoader(dataset, collate_fn=collater, **kwargs)


class DeepIndexDataModule(pl.LightningDataModule):

    def __init__(self, hparams, inference=False):
        super().__init__()
        self.hparams = hparams
        # self.dataset, self.io = process_dataset(self.hparams, inference=inference)

        self.dataset = LazySeqDataset(hparams=self.hparams, keep_open=self.hparams.num_workers==0)
        if self.hparams.load:
            self.dataset.load()
        kwargs = dict(random_state=self.hparams.seed,
                      batch_size=self.hparams.batch_size,
                      pin_memory=False, #self.hparams.pin_memory,
                      shuffle=self.hparams.shuffle,
                      num_workers=self.hparams.num_workers)
        if self.hparams.num_workers > 0:
            kwargs['multiprocessing_context'] = 'spawn'
            kwargs['worker_init_fn'] = self.dataset.worker_init
            kwargs['persistent_workers'] = True

        kwargs.update(self.hparams.loader_kwargs)
        print("------------- DataLoader kwargs:", str(kwargs), file=sys.stderr)
        if inference:
            kwargs['distances'] = False
            kwargs.pop('num_workers', None)
            kwargs.pop('multiprocessing_context', None)
        tr, te, va = train_test_loaders(self.dataset, **kwargs)
        self.loaders = {'train': tr, 'test': te, 'validate': va}

    def train_dataloader(self):
        return self.loaders['train']

    def val_dataloader(self):
        return self.loaders['validate']

    def test_dataloader(self):
        return self.loaders['test']


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
        the target taxonomic level for. This can be 'phylum', 'class', 'order', 'family', 'genus', or
        'species'. For *manifold=True* or *graph=True*, this must be 'species'

    ohe: bool, default=False
        One-hot encode sequences. By default, return indices


    References
    ----------

    PyTorch DataLoaders
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader


    """
    def __init__(self, path=None, keep_open=False, hparams=None, **kwargs):
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

        if hparams is not None:
            if not isinstance(hparams, argparse.Namespace):
                raise ValueError('hparams must be a Namespace object')
            for k, v in vars(hparams).items():
                kwargs[k] = v
        hparams = argparse.Namespace(**kwargs)

        self.path = path or hparams.input
        self.hparams = hparams
        self.load_data = hparams.load

        # memoize any chunking we've done so we only need to do it once
        self._chunkings = dict()

        self.window, self.step = check_window(hparams.window, hparams.step)
        self.revcomp = not hparams.fwd_only

        # open to get dataset length
        self.open()
        self.__len = len(self.difile)
        self.sample_labels = self.difile.labels # taxa_table[hparams.tgt_tax_lvl].data
        self.taxa_labels, self.taxa_counts = np.unique(self.sample_labels, return_counts=True)

        self.vocab_len = len(self.orig_difile.seq_table['sequence'].target.elements)

        #############################
        # self._label_key - the column from the TaxaTable
        #############################

        self.manifold = False
        self.graph = False
        self.distances = None
        self.node_ids = None

        if hparams.manifold:
            self.manifold = True
            self.distances = self.difile.distances.data[:]
            if hparams.tgt_tax_lvl != 'species':
                raise ValueError("must run manifold learning (-M) method with 'species' taxonomic level (-t)")
            self._label_key = 'id'
            self._label_dtype = torch.int64
        elif hparams.graph:
            self.graph = True
            if hparams.tgt_tax_lvl != 'species':
                raise ValueError("must run graph learning (-M) method with 'species' taxonomic level (-t)")
            self._label_key = 'id'
            self._label_dtype = torch.int64

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
            self._label_key = self.hparams.tgt_tax_lvl
            self._label_dtype = torch.int64
        else:
            raise ValueError('classify (-C) or manifold (-M) should be set')

        self.__ohe = hparams.ohe

        if not keep_open:
            self.io.close()
            self.difile = None
            self.orig_difile = None
            self.io = None

    def get_graph(self):
        """Return a csr_matrix representation of the tree graph"""
        return self.difile.tree_graph.to_spmat()

    def open(self):
        """Open the HDMF file and set up chunks and taxonomy label"""
        self.io = get_hdf5io(self.path, 'r')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.orig_difile = self.io.read()
        self.difile = self.orig_difile

        self.difile.set_label_key(self.hparams.tgt_tax_lvl)

        if self.window is not None:
            self.set_chunks(self.window, self.step)
        if self.revcomp:
            self.set_revcomp()
        if self.load_data:
            self.load()

    def worker_init(self, worker_id):
        print("------------- Opening DeepIndexFile for DataLoader worker %s\n\n" % worker_id, file=sys.stderr)
        self.open()

    def set_chunks(self, window, step=None):
        chunks = self._chunkings.get((window, step))
        if chunks is None:
            self._chunkings[(window, step)] = chunk_sequence(self.difile, window, step)
        self.difile = AbstractChunkedDIFile(self.difile, *self._chunkings[(window, step)])

    def set_revcomp(self, revcomp=True):
        self.difile = RevCompFilter(self.difile)

    def set_ohe(self, ohe=True):
        self.__ohe = ohe

    def __len__(self):
        return self.__len # len(self.difile)

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

    def load(self, device=None):
        def _load(data):
            return data[:]

        tfm = self._to_torch(device)
        def to_sint(data):
            return data[:].astype(np.int16)
        self._check_load(self.orig_difile.seq_table['sequence'].target, [to_sint, tfm])
        self._check_load(self.orig_difile.taxa_table[self._label_key], tfm)
        self._check_load(self.orig_difile.distances, tfm)

        for col in self.orig_difile.seq_table.children:
            if col.name == 'sequence':
                continue
            col.transform(_load)

        for col in self.orig_difile.taxa_table.children:
            if col.name == self._label_key:
                continue
            col.transform(_load)

    def __getitem__(self, i):
        # get sequence
        item = self.difile[i]
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
