import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from hdmf.common import get_hdf5io

from ..sequence import WindowChunkedDIFile


def check_window(window, step):
    if window is None:
        return None, None
    else:
        if step is None:
            step = window
        return window, step


def read_dataset(path):
    hdmfio = get_hdf5io(path, 'r')
    difile = hdmfio.read()
    dataset = SeqDataset(difile)
    return dataset, hdmfio

def process_dataset(args, inference=False):
    """
    Process *input* argument and return dataset and HDMFIO object

    Args:
        args (Namespace):       command-line arguments passed by parser
        inference (bool):       load data for inference
    """
    # First, get the dataset, so we can figure
    # out how many outputs there are
    #io = get_hdf5io(args.input, 'r')
    #difile = io.read()
    #dataset = SeqDataset(difile)
    dataset, io = read_dataset(args.input)


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
    args.window, args.step = check_window(args.window, args.step)

    # Process any arguments that impact how we set up the dataset
    if args.window is not None:
        dataset.difile = WindowChunkedDIFile(dataset.difile, args.window, args.step)
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
        maxlen = 0
        l_idx = -1
        for i, X, y in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]
        X_ret = list()
        y_idx = list()
        idx_ret = list()
        size_ret = list()
        for i, X, y in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif))
            X_ret.append(X_)
            size_ret.append(X.shape[l_idx])
            idx_ret.append(i)
            y_idx.append(y)
        X_ret = torch.stack(X_ret)
        y_idx = torch.stack(y_idx)
        # Get distances
        y_ret = self.dmat[y_idx][:, y_idx]
        size_ret = torch.tensor(size_ret)
        idx_ret = torch.tensor(idx_ret)
        return (idx_ret, X_ret, y_ret, size_ret)

def collate(samples):
    """
    A function to collate variable length sequence samples
    """
    maxlen = 0
    l_idx = -1
    for i, X, y in samples:
        if maxlen < X.shape[l_idx]:
            maxlen = X.shape[l_idx]
    X_ret = list()
    y_ret = list()
    idx_ret = list()
    size_ret = list()
    for i, X, y in samples:
        dif = maxlen - X.shape[l_idx]
        X_ = X
        if dif > 0:
            X_ = F.pad(X, (0, dif))
        X_ret.append(X_)
        y_ret.append(y)
        size_ret.append(X.shape[l_idx])
        idx_ret.append(i)
    X_ret = torch.stack(X_ret)
    y_ret = torch.stack(y_ret)
    size_ret = torch.tensor(size_ret)
    idx_ret = torch.tensor(idx_ret)
    return (idx_ret, X_ret, y_ret, size_ret)


class SeqDataset(Dataset):
    """
    A torch Dataset to handle reading samples read from a DeepIndex file
    """

    def __init__(self, difile, classify=False):
        self.difile = difile

        self.one_hot = self.get_one_hot(True)

        self.set_classify(classify)
        self._target_key = 'class_label' if classify else 'embedding'


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
        tfm = self._to_torch(device)
        def to_sint(data):
            return data[:].astype(np.int16)
        self._check_load(self.difile.seq_table['sequence'].target, [to_sint, tfm])
        self._check_load(self.difile.taxa_table[self._label_key], tfm)

    def __getitem__(self, i):
        # get sequence
        idx, seq, label = self.difile[i]
        ## one-hot encode sequence
        seq = F.one_hot(torch.as_tensor(seq, dtype=torch.int64)).float().T
        label = torch.as_tensor(label, dtype=self._label_dtype)
        return (idx, seq, label)

    @staticmethod
    def get_one_hot(torch=True):
        if torch:
            return lambda seq: F.one_hot(seq.long()).float()
        else:
            return lambda seq: np.eye(np.max(seq)+1)[seq]


def get_loader(path, **kwargs):
    """
    Return a DataLoader that loads data from the given DeepIndex file

    Args:
        path (str): the path to the DeepIndex file
        kwargs    : any additional arguments to pass into torch.DataLoader
    """
    hdmfio = get_hdf5io(path, 'r')
    loader = DataLoader(SeqDataset(hdmfio), collate_fn=collate, **kwargs)
    return loader


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

    train_dataset = DatasetSubset(dataset, train_idx)
    test_dataset = DatasetSubset(dataset, test_idx)
    validate_dataset = DatasetSubset(dataset, validate_idx)
    return (DataLoader(train_dataset, collate_fn=collater, **kwargs),
            DataLoader(test_dataset, collate_fn=collater, **kwargs),
            DataLoader(validate_dataset, collate_fn=collater, **kwargs))
