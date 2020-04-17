import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class RozNet(nn.Module):
    '''
    A 1D CNN with 5 convolutional layers, followed by 3 fully-connected layers

    Args:
        input_nc (int):  the input number of channels
    '''

    def __init__(self, input_nc, n_outputs=2, first_kernel_size=7, maxpool=False):
        super(RozNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_nc, 64, kernel_size=first_kernel_size, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        if maxpool:
            self.pool = nn.AdaptiveMaxPool1d(24)
        else:
            self.pool = nn.AdaptiveAvgPool1d(24)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*24, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_outputs),
        )

    def forward(self, x, **kwargs):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    from .train import parse_args, run_serial, check_model, load_dataset
    import torch.optim as optim

    args = parse_args("Train CNN with Spatial Pyramidal Pooling")
                      #[['-E', '--emb_nc'], dict(type=int, default=0, help='the number of embedding channels. default is no embedding')])

    input_nc = 4
    if args['protein']:
        input_nc = 26

    if args['sanity']:
        input_nc = 5


    dataset, io = load_dataset(path=args['input'], **args)
    breakpoint()

    model = check_model(RozNet(input_nc, n_outputs=dataset.difile.n_emb_components), **args)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])


    args['pad'] = True

    run_serial(dataset=dataset, model=model, optimizer=optimizer, **args)
    io.close()
