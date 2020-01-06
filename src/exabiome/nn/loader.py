import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from hdmf.common import get_hdf5io


def collate(samples):
    """
    A function to collate variable length sequence samples
    """
    maxlen = 0
    for i, X, y in samples:
        if maxlen < X.shape[1]:
            maxlen = X.shape[1]
    X_ret = list()
    y_ret = list()
    idx_ret = list()
    size_ret = list()
    for i, X, y in samples:
        dif = maxlen - X.shape[1]
        X_ = X
        if dif > 0:
            X_ = F.pad(X, (0, dif))
        X_ret.append(X_)
        y_ret.append(y)
        size_ret.append(X.shape[1])
        idx_ret.append(i)
    X_ret = torch.stack(X_ret).float()
    y_ret = torch.stack(y_ret).float()
    size_ret = torch.tensor(size_ret)
    return (idx_ret, X_ret, y_ret, size_ret)


class SeqDataset(Dataset):
    """
    A torch Dataset to handle reading samples read from a DeepIndex file
    """

    def __init__(self, hdmfio, device=None, **kwargs):
        self.hdmfio = hdmfio
        self.difile = self.hdmfio.read()
        self.device = device
        self.difile.set_torch(True, dtype=torch.float, device=device)

    def __len__(self):
        return len(self.difile)

    def close(self):
        self.hdmfio.close()

    def __getitem__(self, i):
        d = self.difile[i]
        return i, d['sequence'], torch.from_numpy(d['embedding'])


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

def train_test_loaders(path, random_state=None, test_size=None, train_size=None,
                       stratify=None, device=None, **kwargs):
    """
    Return DataLoaders for training and test datasets.

    Args:
        path (str): the path to the DeepIndex file
        kwargs    : any additional arguments to pass into torch.DataLoader
    """

    hdmfio = get_hdf5io(path, 'r')
    difile = hdmfio.read()
    train_idx, test_idx = train_test_split(np.arange(len(difile.seq_table)),
                                           random_state=random_state,
                                           train_size=train_size,
                                           test_size=test_size,
                                           stratify=difile.seq_table['taxon'].data[:])
    dataset = SeqDataset(hdmfio, device=device)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    return (DataLoader(dataset, collate_fn=collate, sampler=train_sampler, **kwargs),
            DataLoader(dataset, collate_fn=collate, sampler=test_sampler, **kwargs))
