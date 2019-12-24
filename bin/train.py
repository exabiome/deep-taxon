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
parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

loglvl = logging.DEBUG if args.debug else logging.INFO

logging.basicConfig(stream=sys.stdout, level=loglvl, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

logger.info('loading data %s' % args.input)
train_loader, test_loader = train_test_loaders(args.input, test_size=0.2, batch_size=args.batch_size)

input_nc = 4
if args.protein:
    input_nc = 26

model = SPP_CNN(input_nc, 100, kernel_size=13)
optimizer = optim.Adam(model.parameters(), lr=0.001)

curr_epoch = 0
n_epochs = args.epochs

train_loss = np.zeros(n_epochs)
test_loss = np.zeros(n_epochs)
if args.checkpoint:
    logger.info('picking up from checkpoint %s' % args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    curr_epoch = checkpoint['epoch']
    train_loss = np.append(checkpoint['train_loss'], np.zeros(n_epochs))
    test_loss = np.append(checkpoint['test_loss'], np.zeros(n_epochs))
    logger.info('beginning at epoch %d' % curr_epoch)

last_epoch = curr_epoch + n_epochs

criterion = nn.MSELoss()
before = datetime.now()

if args.debug:
    n_epochs = 1

for curr_epoch in range(curr_epoch, last_epoch):  # loop over the dataset multiple times

    prev_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        idx, seqs, emb, orig_lens = data

        # zero the parameter gradients
        optimizer.zero_grad()

        ## Modify here to adjust learning rate
        ## Create new Optimizer object?

        # forward + backward + optimize
        logger.debug('forward')
        outputs = model(seqs, orig_len=orig_lens)

        logger.debug('criterion')
        loss = criterion(outputs, emb)
        logger.debug('backward')
        loss.backward()
        logger.debug('step')
        optimizer.step()

        train_loss[curr_epoch] += loss.item()
        if i % 40 == 39:    # print every 2000 mini-batches
            logger.info('[%d, %5d] loss: %.6e' %
                        (curr_epoch + 1, i + 1, (train_loss[curr_epoch]-prev_loss)/40))
            prev_loss = train_loss[curr_epoch]
        if args.debug:
            if i == 1:
                break

    logger.debug('test loss')
    for i, data in enumerate(test_loader, 0):
        idx, seqs, emb, orig_lens = data
        outputs = model(seqs, orig_len=orig_lens)
        test_loss[curr_epoch] += criterion(outputs, emb).item()
        if args.debug:
            break

if args.output is not None:
    logger.info('saving to %s' % args.output)
    torch.save({'epoch': last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': train_loss,
                'loss': loss,
                }, args.output)

after = datetime.now()
logger.info('Finished Training. Took %s seconds' % (after-before).total_seconds())
