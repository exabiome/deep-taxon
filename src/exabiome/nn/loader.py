import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from hdmf.common import get_hdf5io


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
    return (idx_ret, X_ret, y_ret, size_ret)


class SeqDataset(Dataset):
    """
    A torch Dataset to handle reading samples read from a DeepIndex file
    """

    def __init__(self, difile, device=None, classify=False, **kwargs):
        self.difile = difile
        self.device = device
        #self.difile.set_torch(True, dtype=torch.long, device=device,
        #                      ohe=kwargs.get('ohe', True),
        #                      pad=kwargs.get('pad', False))

        self.set_classify(classify)
        #self.difile.seq_table.set_torch(True, dtype=torch.long, device=self.device)
        #self.difile.set_sanity(kwargs.get('sanity', False))
        self._target_key = 'class_label' if classify else 'embedding'

    def set_classify(self, classify):
        self._classify = classify
        #if classify:
        #    self.difile.label_key = 'id'
        #    self.difile.taxa_table.set_torch(True, dtype=torch.long, device=self.device)
        #else:
        #    self.difile.label_key = 'embedding'
        #    self.difile.taxa_table.set_torch(True, dtype=torch.float, device=self.device)

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

    def load(self, torch=False, device=None):
        tfm = self._to_torch(device)
        def to_sint(data):
            return data[:].astype(np.int16)
        self.difile.seq_table['taxon'].transform(to_sint).transform(tfm)
        self.difile.seq_table['sequence'].target.transform(to_sint).transform(tfm)
        if not self._classify:
            label = self.difile.taxa_table['embedding'].transform(tfm)

    def __getitem__(self, i):
        # get sequence index
        idx = self.difile.seq_table.id[i]

        # get sequence, and one-hot encode
        seq = self.difile.seq_table['sequence'].get(i, index=True)
        seq = F.one_hot(seq.long()).T.float()

        # get label
        label = self.difile.seq_table['taxon'].get(i, index=True).long()
        if not self._classify:
            label = self.difile.taxa_table['embedding'][label]
        return (idx, seq, label)

        #d = self.difile[i]
        #return d[0], F.one_hot(d[1]).T.float(), d[2]


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


def train_test_loaders(dataset, random_state=None, downsample=None,
                       **kwargs):
    """
    Return DataLoaders for training and test datasets.

    Args:
        path (str): the path to the DeepIndex file
        kwargs    : any additional arguments to pass into torch.DataLoader
    """
    index = np.arange(len(dataset))
    stratify = dataset.difile.labels
    if downsample is not None:
        index, _, stratify, _ = train_test_split(index, stratify, train_size=downsample)

    train_idx, test_idx, validate_idx = train_test_validate_split(index,
                                                                  stratify=stratify,
                                                                  random_state=random_state)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    validate_sampler = SubsetRandomSampler(validate_idx)
    return (DataLoader(dataset, collate_fn=collate, sampler=train_sampler, **kwargs),
            DataLoader(dataset, collate_fn=collate, sampler=test_sampler, **kwargs),
            DataLoader(dataset, collate_fn=collate, sampler=validate_sampler, **kwargs))
