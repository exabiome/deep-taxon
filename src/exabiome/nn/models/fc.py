import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from . import model, AbstractLit

@model('fc')
class FC(AbstractLit):
    '''
    A 1D CNN with 5 convolutional layers, followed by 3 fully-connected layers

    Args:
        input_nc (int):  the input number of channels
    '''

    def __init__(self, input_nc):
        super(FC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_nc, input_nc),
            nn.Linear(input_nc, 2),
        )

    def forward(self, x, **kwargs):
        x = torch.flatten(x, 1)
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
