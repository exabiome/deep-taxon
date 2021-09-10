import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from . import model, AbstractLit


def fc_layer(in_features, out_features, dropout=0.5, relu=True, batch_norm=True):
    layers = [
        nn.Dropout(p=dropout),
        nn.Linear(in_features, out_features),
    ]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


@model('mlp')
class MLP(AbstractLit):
    '''
    A multi-layer perceptron for classifying species using tetranucleotide frequency
    '''

    def __init__(self, hparams):
        super(MLP, self).__init__()
        outputs = list(hparams.outputs)

        in_features = hparams.input_nc        # for TNF, this should be 136
        layers = list()
        for out_features in outputs:
            layers.append(fc_layer(in_features, out_features))
            in_features = out_features
        layers.append(fc_layer(in_features, hparams.n_outputs, relu=False, batch_norm=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x



if __name__ == '__main__':
    from .train import parse_args, run_serial
    import torch.optim as optim

    args = parse_args("Train CNN with Spatial Pyramidal Pooling")
                      #[['-E', '--emb_nc'], dict(type=int, default=0, help='the number of embedding channels. default is no embedding')])

    input_nc = 4
    if args['protein']:
        input_nc = 26

    if args['sanity']:
        input_nc = 25

    model = FC(input_nc)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    args['pad'] = True

    run_serial(model=model, optimizer=optimizer, **args)
