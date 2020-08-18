import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from . import model, AbstractLit


@model('roznet')
class RozNet(AbstractLit):
    '''
    A 1D CNN with 5 convolutional layers, followed by 3 fully-connected layers

    Args:
        input_nc (int):  the input number of channels
    '''

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.check_hparams(hparams)
        input_nc = getattr(hparams, 'input_nc', None)
        n_outputs = getattr(hparams, 'n_outputs', 2)
        first_kernel_size = getattr(hparams, 'first_kernel_size', 7)
        maxpool = getattr(hparams, 'maxpool', True)
        emb_dim = 8
        self.embedding = nn.Embedding(input_nc, emb_dim)
        self.features = nn.Sequential(
            nn.Conv1d(emb_dim, 64, kernel_size=first_kernel_size, stride=1, padding=2),
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
        self.classifier = nn.Sequential(
            nn.Dropout(hparams.dropout_rate),
            nn.Linear(256*pool_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(hparams.dropout_rate),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_outputs),
            #nn.BatchNorm1d(n_outputs)
        )

    def forward(self, x, **kwargs):
<<<<<<< HEAD
        print("===1", x.shape)
        x = self.embedding(x).permute(0, 2, 1)
        print("===2", x.shape)
=======
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
>>>>>>> upstream/master
        x = self.features(x)
        print("===3", x.shape)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        print("===4", x.shape)
        x = self.classifier(x)
        print("====5", x.shape)
        return x



if __name__ == '__main__':
    from .train import parse_args, run, check_model, load_dataset

    args = parse_args("Train CNN with Spatial Pyramidal Pooling")
    #[['-E', '--emb_nc'], dict(type=int, default=0, help='the number of embedding channels. default is no embedding')])

    input_nc = 5
    if args['protein']:
        input_nc = 26

    if args['sanity']:
        input_nc = 5


    dataset, io = load_dataset(path=args['input'], **args)

    if args['classify']:
        n_outputs = len(dataset.difile.taxa_table)
    else:
        n_outputs = dataset.difile.n_emb_components

    model = check_model(RozNet(input_nc, n_outputs=n_outputs), **args)

    args['pad'] = True

    run(dataset, model, **args)

    io.close()
