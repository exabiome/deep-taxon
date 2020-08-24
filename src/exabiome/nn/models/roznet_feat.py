import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from . import model, AbstractLit


@model('roznet_feat')
class RozNetFeat(AbstractLit):
    '''RozNet with the fully-connected layers removed
    '''

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.check_hparams(hparams)
        input_nc = getattr(hparams, 'input_nc', None)
        n_outputs = getattr(hparams, 'n_outputs', 2)
        first_kernel_size = getattr(hparams, 'first_kernel_size', 7)
        maxpool = getattr(hparams, 'maxpool', True)
        self.embedding = nn.Embedding(input_nc, 8)
        self.features = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=first_kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )
        pool_size = 24
        if maxpool:
            self.pool = nn.AdaptiveMaxPool1d(pool_size)
        else:
            self.pool = nn.AdaptiveAvgPool1d(pool_size)

    def forward(self, x, **kwargs):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
