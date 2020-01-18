import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class SpatialPyramidPool1d(nn.Module):
    """
    A 1D spatial-pyramidal pooling layer
    """

    def __init__(self, num_levels, shift=0, pool_type='max_pool'):
        super(SpatialPyramidPool1d, self).__init__()
        self.num_levels = num_levels
        # the shift in sample length as
        # a result of doing convolutions
        self.shift = shift
        if pool_type == 'max_pool':
            self.pool = F.max_pool1d
        elif pool_type == 'avg_pool':
            self.pool = F.avg_pool1d
        else:
            raise ValueError('unrecognized pool_type: %s' % pool_type)

    def _pool_ragged(self, x, orig_len):
        ret = torch.zeros([x.shape[0], x.shape[1]*(2**self.num_levels - 1)], device=x.device)
        for x_i in range(x.shape[0]):
            sample = x[x_i, :, :orig_len[x_i] - self.shift].unsqueeze(0)
            f = sample.shape[2]
            s = 0
            e = sample.shape[1]
            for i in range(self.num_levels):
                ret[x_i, s:s + e] = self.pool(sample,
                                              kernel_size=math.ceil(f),
                                              stride=math.floor(f)).view(e)
                s += e
                e *= 2
                f /= 2
        return ret

    def forward(self, x, orig_len=None):
        if orig_len is None:
            orig_len = torch.ones(x.shape[0]) * x.shape[2]
        return self._pool_ragged(x, orig_len)


class SPP_CNN(nn.Module):
    '''
    A CNN model which adds SPP layer so that we can input variable length tensors

    Args:
        input_nc (int):  the input number of channels
    '''
    def __init__(self, input_nc, output_nc=None, n_levels=2, n_tasks=2, kernel_size=21, emb_nc=0):
        super(SPP_CNN, self).__init__()
        n_lin = 0
        if output_nc is None:
            output_nc = input_nc

        self.embedding = None
        if emb_nc > 0:
            print('setting Embedding')
            self.embedding = nn.Embedding(input_nc, emb_nc)
            input_nc = emb_nc

        self.conv1 = nn.Conv1d(input_nc, output_nc, kernel_size,
                               stride=1,
                               padding=0,
                               dilation=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(output_nc)

        self.spp1 = SpatialPyramidPool1d(n_levels, kernel_size-1)
        n_lin += self.conv1.out_channels * (2**n_levels - 1)

        self.conv2 = nn.Conv1d(input_nc, output_nc, kernel_size,
                               stride=1,
                               padding=0,
                               dilation=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(output_nc)
        self.spp2 = SpatialPyramidPool1d(n_levels, kernel_size-1, pool_type='avg_pool')
        n_lin += self.conv2.out_channels * (2**n_levels - 1)

        self.fc1 = nn.Linear(n_lin, n_tasks)

    def forward(self, x, orig_len=None):

        if self.embedding is not None:
            x = self.embedding(x).permute(0, 2, 1)

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.spp1(x1, orig_len=orig_len)

        x2 = x
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.spp2(x2, orig_len=orig_len)

        xf = self.fc1(torch.cat([x1, x2], dim=1))

        return xf


class GroupPool(nn.Module):
    """
    This might be unnecessary

    I think pooling by taxa will discard intra-genome variation
    """
    def __init__(self, pool_type='avg_pool'):
        super(GroupPool, self).__init__()
        if pool_type == 'avg_pool':
            self.pool = self.__mean
        elif pool_type == 'max_pool':
            self.pool = self.__max
        else:
            raise ValueError("Unrecognized pooling option: %s" % pool_type)

    @staticmethod
    def __max(x):
        return torch.max(x, axis=1)[0]

    @staticmethod
    def __mean(x):
        return torch.mean(x, axis=1)

    def _pool_groups(self, x, group):
        groups = torch.unique(group)
        ret = torch.zeros([groups.shape[0], x.shape[1]])
        for g_i, g in enumerate(groups):
            mask = group == g
            ret[g_i] = self.pool(x[mask])
        return ret

    def forward(self, x, group=None):
        return self._pool_groups(x, group)
