import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import exabiome.sequence.dna_table
from hdmf.common import get_hdf5io


def collate(samples):
    """
    A function to collate variable length sequence samples
    """
    maxlen = 0
    for X, y in samples:
        if maxlen < X.shape[1]:
            maxlen = X.shape[1]
    X_ret = list()
    y_ret = list()
    size_ret = list()
    for X, y in samples:
        dif = maxlen - X.shape[1]
        X_ = X
        if dif > 0:
            X_ = F.pad(X, (0, dif))
        X_ret.append(X_)
        y_ret.append(y)
        size_ret.append(X.shape[1])
    X_ret = torch.stack(X_ret).float()
    y_ret = torch.stack(y_ret).float()
    size_ret = torch.tensor(size_ret)
    return (X_ret, y_ret, size_ret)


class DNADataset(Dataset):
    """
    A torch Dataset to handle reading samples read from a DeepIndex file
    """

    def __init__(self, path):
        self.hdmfio = get_hdf5io(path, 'r')
        self.difile = self.hdmfio.read()

    def __len__(self):
        return len(self.difile)

    def __getitem__(self, i):
        d = self.difile[i]
        return torch.from_numpy(d['sequence']), torch.from_numpy(d['embedding'])

    def close(self):
        self.hdmfio.close()


def get_loader(path, **kwargs):
    """
    Return a DataLoader that loads data from the given DeepIndex file

    Args:
        path (str): the path to the DeepIndex file
        kwargs    : any additional arguments to pass into torch.DataLoader
    """
    loader = DataLoader(DNADataset(path), collate_fn=collate, **kwargs)
