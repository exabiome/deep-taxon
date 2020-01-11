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

def parse_seed(string):
    if string is None:
        return int(datetime.now().timestamp())
    else:
        return int(string)

def parse_test_size(string):
    ret = float(string)
    if ret > 1.0:
        ret = int(ret)
    return ret

epi = """
output can be used as a checkpoint
"""

parser = argparse.ArgumentParser(epilog=epi)
parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
parser.add_argument('output', type=str, help='file to save model', default=None)
parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
parser.add_argument('-p', '--protein', action='store_true', default=False, help='input contains protein sequences')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
parser.add_argument('-g', '--gpu', action='store_true', default=False, help='use GPU')
parser.add_argument('-s', '--seed', type=parse_seed, default=None, help='seed to use for train-test split')
parser.add_argument('-t', '--test_size', type=parse_test_size, default=0.2, help='size of test split')


if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

split_seed = args.seed if args.seed is not None else int(datetime.now().timestamp())
test_size = args.test_size
batch_size = args.batch_size

loglvl = logging.DEBUG if args.debug else logging.INFO

logging.basicConfig(stream=sys.stdout, level=loglvl, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

input_nc = 4
if args.protein:
    input_nc = 26

model = SPP_CNN(input_nc, 100, kernel_size=13)

device = None
if args.gpu:
    device = torch.device("cuda:0")
    model.to(device)
    #optimizer.to(device)

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
    test_size = checkpoint['test_size']
    split_seed = checkpoint['split_seed']
    batch_size = checkpoint['batch_size']

last_epoch = curr_epoch + n_epochs
criterion = nn.MSELoss()

logger.info('loading data from %s' % args.input)
logger.info(f'- using {split_seed} as seed for train-test split')
logger.info(f'- test size: {test_size}')
logger.info(f'- batch size: {batch_size}')
train_loader, test_loader = train_test_loaders(args.input,
                                               test_size=test_size,
                                               batch_size=batch_size,
                                               device=device,
                                               random_state=split_seed)

logger.info(f'starting with epoch {curr_epoch+1}')
logger.info(f'saving results to {args.output}')

nprint = len(train_loader.dataset) // 10

before = datetime.now()
for curr_epoch in range(curr_epoch, last_epoch):  # loop over the dataset multiple times
    logger.info(f'begin epoch {curr_epoch+1}')

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
        emb = emb.to(device)
        loss = criterion(outputs, emb)
        logger.debug('backward')
        loss.backward()
        logger.debug('step')
        optimizer.step()

        train_loss[curr_epoch] += loss.item()
        if i % nprint == nprint - 1:    # print every 10th of training set
            logger.info('[%d, %5d] loss: %.6e' %
                        (curr_epoch + 1, i + 1, (train_loss[curr_epoch]-prev_loss)/nprint))
            prev_loss = train_loss[curr_epoch]
        if args.debug:
            if i == 1:
                break

    logger.debug('test loss')
    for i, data in enumerate(test_loader, 0):
        idx, seqs, emb, orig_lens = data
        emb = emb.to(device)
        outputs = model(seqs, orig_len=orig_lens)
        crit = criterion(outputs, emb).item()
        test_loss[curr_epoch] += crit
        if args.debug:
            break

    logger.debug('checkpointing')
    torch.save({'epoch': last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': train_loss,
                'split_seed': split_seed,
                'test_size': test_size,
                'batch_size': batch_size,
                'loss': loss,
                }, args.output)

    logger.info(f'epoch {curr_epoch+1} complete')

after = datetime.now()
logger.info('Finished Training. Took %s seconds' % (after-before).total_seconds())
