import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from hdmf.common import get_hdf5io

from ..sequence import WindowChunkedDIFile, RevCompFilter
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
    group.add_argument('-F', '--fwd_only', default=False, action='store_true', help='use forward strand of sequences only')
    type_group = group.add_mutually_exclusive_group()
    type_group.add_argument('-C', '--classify', action='store_true', help='run a classification problem', default=False)
    type_group.add_argument('-M', '--manifold', action='store_true', help='run a manifold learning problem', default=False)

    return None


def dataset_stats(argv=None):
    """Read a dataset and print the number of samples to stdout"""

    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    add_dataset_arguments(parser)
    test_group = parser.add_argument_group('Test reading')
    test_group.add_argument('-T', '--test_read', default=False, action='store_true', help='test reading an element')
    test_group.add_argument('-s', '--seed', type=parse_seed, default=None, help='seed for an 80/10/10 split before reading an element')
    test_group.add_argument('-l', '--load', action='store_true', default=False, help='load data into memory before running training loop')

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

def read_dataset(path):
    hdmfio = get_hdf5io(path, 'r')
    difile = hdmfio.read()
    dataset = SeqDataset(difile)
    return dataset, hdmfio


def process_dataset(args, path=None, inference=False):
    """
    Process *input* argument and return dataset and HDMFIO object
    Args:
        args (Namespace):       command-line arguments passed by parser
        inference (bool):       load data for inference
    """
    dataset, io = read_dataset(path or args.input)

    if not hasattr(args, 'classify'):
        raise ValueError('Parser must check for classify/regression/manifold '
                         'to determine the number of outputs')
    if args.classify:
        dataset.set_classify(True)
        n_outputs = len(dataset.difile.taxa_table)
    elif args.manifold:
        dataset.set_classify(True)
        n_outputs = 32        #TODO make this configurable #breakpoint
    else:
        args.regression = True
        dataset.set_classify(False)
    if inference:
        dataset.set_classify(True)
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

    def __init__(self, difile, classify=False):
        self.difile = difile
        self.set_classify(classify)
        self._target_key = 'class_label' if classify else 'embedding'
        self.vocab_len = len(self.difile.seq_table['sequence'].target.vocabulary)
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

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)


def train_test_loaders(dataset, random_state=None, downsample=None, distances=False,
                       **kwargs):
    """
    Return DataLoaders for training and test datasets.
    Args:
        path (str):                  the path to the DeepIndex file
        distances (str):             return distances for the
        kwargs    : any additional arguments to pass into torch.DataLoader
    """
    index = np.arange(len(dataset))
    stratify = dataset.difile.labels
    if downsample is not None:
        index, _, stratify, _ = train_test_split(index, stratify,
                                                 train_size=downsample,
                                                 random_state=random_state)

    train_idx, test_idx, validate_idx = train_test_validate_split(index,
                                                                  stratify=stratify,
                                                                  random_state=random_state)

    collater = collate
    if distances:
        collater = DistanceCollater(dataset.difile.distances.data[:])

    kwargs['pin_memory'] = True

    train_dataset = DatasetSubset(dataset, train_idx)
    test_dataset = DatasetSubset(dataset, test_idx)
    validate_dataset = DatasetSubset(dataset, validate_idx)
    return (DataLoader(train_dataset, collate_fn=collater, **kwargs),
            DataLoader(test_dataset, collate_fn=collater, **kwargs),
            DataLoader(validate_dataset, collate_fn=collater, **kwargs))

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
        self.dataset, self.io = process_dataset(self.hparams, inference=inference)
        if self.hparams.load:
            self.dataset.load()
        kwargs = dict(random_state=self.hparams.seed,
                      batch_size=self.hparams.batch_size,
                      distances=self.hparams.manifold)
        kwargs.update(self.hparams.loader_kwargs)
        if inference:
            kwargs['distances'] = False
            kwargs.pop('num_workers', None)
            kwargs.pop('multiprocessing_context', None)
        tr, te, va = train_test_loaders(self.dataset, **kwargs)
        self.loaders = {'train': tr, 'test': te, 'validate': va}

        if self.hparams.classify:
            self.n_outputs = len(self.dataset.difile.taxa_table)
        else:
            self.n_outputs = self.hparams.n_outputs

    def train_dataloader(self):
        return self.loaders['train']

    def val_dataloader(self):
        return self.loaders['validate']

    def test_dataloader(self):
        return self.loaders['test']
