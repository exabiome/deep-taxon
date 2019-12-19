import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
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


class SeqDataset(Dataset):
    """
    A torch Dataset to handle reading samples read from a DeepIndex file
    """

    def __init__(self, hdmfio):
        self.hdmfio = hdmfio
        self.difile = self.hdmfio.read()

    def __len__(self):
        return len(self.difile)

    def close(self):
        self.hdmfio.close()


class DNADataset(SeqDataset):
    """
    A torch Dataset to handle reading DNA samples read from a DeepIndex file
    """

    def __getitem__(self, i):
        d = self.difile[i]
        return torch.from_numpy(d['sequence']), torch.from_numpy(d['embedding'])


class AADataset(SeqDataset):
    """
    A torch Dataset to handle reading protein samples read from a DeepIndex file
    """

    def __getitem__(self, i):
        d = self.difile[i]
        ohe_pos = d['sequence']
        tensor = torch.zeros((ohe_pos.shape[0], 26))
        tensor[np.arange(ohe_pos.shape[0]), ohe_pos] = 1
        return tensor, torch.from_numpy(d['embedding'])


def get_loader(path, **kwargs):
    """
    Return a DataLoader that loads data from the given DeepIndex file

    Args:
        path (str): the path to the DeepIndex file
        kwargs    : any additional arguments to pass into torch.DataLoader
    """
    hdmfio = get_hdf5io(path, 'r')
    loader = DataLoader(DNADataset(hdmfio), collate_fn=collate, **kwargs)
    return loader
