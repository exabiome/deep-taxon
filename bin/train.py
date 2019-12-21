import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import exabiome.sequence
from exabiome.nn import train_test_loaders
from exabiome.nn.model import SPP_CNN

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
parser.add_argument('-o', '--output', type=str, help='file to save model', default=None)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
parser.add_argument('-p', '--protein', action='store_true', default=False, help='file contains protein sequences')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

logger.info('loading data %s' % args.input)
train_loader, test_loader = train_test_loaders(args.input, test_size=0.2, batch_size=args.batch_size)

input_nc = 4
if args.protein:
    input_nc = 26

model = SPP_CNN(input_nc, 100, kernel_size=13)
optimizer = optim.Adam(model.parameters(), lr=0.001)

if args.checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

criterion = nn.MSELoss()

n_epochs = 1

before = datetime.now()

train_loss = np.zeros(n_epochs)
test_loss = np.zeros(n_epochs)

epoch = 0
for epoch in range(n_epochs):  # loop over the dataset multiple times

    prev_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        idx, seqs, emb, orig_lens = data

        # zero the parameter gradients
        optimizer.zero_grad()

        ## Modify here to adjust learning rate
        ## Create new Optimizer object?

        # forward + backward + optimize
        outputs = model(seqs, orig_len=orig_lens)

        loss = criterion(outputs, emb)
        loss.backward()
        optimizer.step()

        train_loss[epoch] += loss.item()
        if i % 40 == 39:    # print every 2000 mini-batches
            logger.info('[%d, %5d] loss: %.6e' %
                        (epoch + 1, i + 1, (train_loss[epoch]-prev_loss)/40))
            prev_loss = train_loss[epoch]

    for i, data in enumerate(test_loader, 0):
        idx, seqs, emb, orig_lens = data
        outputs = model(seqs, orig_len=orig_lens)
        test_loss[epoch] += criterion(outputs, emb).item()


if args.output is not None:
    logger.info('saving to %s' % args.output)
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, args.output)

after = datetime.now()
logger.info('Finished Training. Took %s seconds' % (after-before).total_seconds())
