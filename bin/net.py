import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPool1d(nn.Module):
    def __init__(self, num_levels, shift=0, pool_type='max_pool'):
        super(SpatialPyramidPool1d, self).__init__()
        self.num_levels = num_levels
        self.shift = shift            # the shift in sample length as
                                      # a result of doing convolutions
        if pool_type == 'max_pool':
            self.pool = F.max_pool1d
        else:
            self.pool = F.avg_pool1d

    def _pool_ragged(self, x, orig_len):
        ret = torch.zeros([x.shape[0], x.shape[1]*(2**self.num_levels - 1)])
        for x_i in range(x.shape[0]):
            sample = x[x_i,:,:orig_len[x_i]-self.shift].unsqueeze(0)
            layers = list()
            f = sample.shape[2]
            s = 0
            e = sample.shape[1]
            for i in range(self.num_levels):
                ret[x_i,s:s+e] = self.pool(sample,
                                           kernel_size=math.ceil(f),
                                           stride=math.floor(f)).view(e)

                s += e
                e *= 2
                f /= 2
        return ret


    def forward(self, x, orig_len=None):
        return self._pool_ragged(x, orig_len)


class SPP_CNN(nn.Module):
    '''
    A CNN model which adds spp layer so that we can input multi-size tensor
    '''
    def __init__(self, input_nc, n_levels=2, n_tasks=2, kernel_size=21):
        super(SPP_CNN, self).__init__()
        ndf = input_nc
        n_lin = 0
        self.conv1 = nn.Conv1d(input_nc, input_nc, kernel_size,
                               stride=1,
                               padding=0,
                               dilation=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(input_nc)
        self.spp1 = SpatialPyramidPool1d(n_levels, kernel_size-1)
        n_lin += self.conv1.out_channels * (2**n_levels - 1)

        self.conv2 = nn.Conv1d(input_nc, input_nc, kernel_size,
                               stride=1,
                               padding=0,
                               dilation=2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(input_nc)
        self.spp2 = SpatialPyramidPool1d(n_levels, 2*(kernel_size-1))
        n_lin += self.conv2.out_channels * (2**n_levels - 1)

        self.fc1 = nn.Linear(n_lin, n_tasks)

    def forward(self, x, orig_len=None):
        x1 = self.conv1(x)
        x1 = F.leaky_relu(self.bn1(x1))
        x1 = self.spp1(x1, orig_len=orig_len)

        x2 = self.conv1(x)
        x2 = F.leaky_relu(self.bn1(x2))
        x2 = self.spp2(x2, orig_len=orig_len)

        xf = self.fc1(torch.cat([x1, x2], dim=1))
        return xf
