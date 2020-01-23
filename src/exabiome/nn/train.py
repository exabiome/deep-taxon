import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import exabiome.sequence
from exabiome.nn import train_test_loaders
from hdmf.utils import docval

import argparse
import logging


def parse_seed(string):
    if string:
        try:
            i = int(string)
            if i > 2**32 - 1:
                raise ValueError(string)
            return i
        except :
            raise argparse.ArgumentTypeError(f'{string} is not a valid seed')
    else:
        return int(datetime.now().timestamp())


def parse_train_size(string):
    ret = float(string)
    if ret > 1.0:
        ret = int(ret)
    return ret


def parse_logger(string):
    if not string:
        ret = logging.getLogger('stdout')
        hdlr = logging.StreamHandler(sys.stdout)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)
    ret.setLevel(logging.INFO)
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return ret


def _parse_cuda_index_helper(s):
    try:
        i = int(s)
        if i > torch.cuda.device_count() or i < 0:
            raise ValueError(s)
        return i
    except :
        devices = str(np.arange(torch.cuda.device_count()))
        raise argparse.ArgumentTypeError(f'{s} is not a valid CUDA index. Please choose from {devices}')


def parse_cuda_index(string):
    if string == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        if ',' in string:
            return [_parse_cuda_index_helper(_) for _ in string.split(',')]
        else:
            return _parse_cuda_index_helper(string)



def parse_args(desc, *addl_args, argv=None):
    """
    Parse arguments for training executable
    """
    if argv is None:
        argv = sys.argv[1:]

    epi = """
    output can be used as a checkpoint
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('output', type=str, help='file to save model', default=None)
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-r', '--resume', action='store_true', help='resume training from checkpoint stored in output', default=False)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
    parser.add_argument('-p', '--protein', action='store_true', default=False, help='input contains protein sequences')
    parser.add_argument('-g', '--gpu', action='store_true', default=False, help='use GPU')
    parser.add_argument('-C', '--cuda_index', type=parse_cuda_index, default='all', help='which CUDA device to use')
    parser.add_argument('-s', '--split_seed', type=parse_seed, default='', help='seed to use for train-test split')
    parser.add_argument('-t', '--train_size', type=parse_train_size, default=0.8, help='size of train split')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('-D', '--downsample', type=float, default=None, help='downsample input before training')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('--prof', type=str, default=None, metavar='PATH', help='profile training loop dump results to PATH')
    parser.add_argument('--sanity', action='store_true', default=False, help='copy response data into input data')
    parser.add_argument('-L', '--load', action='store_true', default=False, help='load data into memory before running training loop')
    parser.add_argument('--lr', type=float, default=0.01, help='the learning rate for Adam')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    ret = vars(args)
    if args.checkpoint is None:
        ret['checkpoint'] = False
    if args.resume:
        if not isinstance(ret['checkpoint'], str):
            # don't overwrite a given path
            ret['checkpoint'] = True
    ret.pop('resume')
    return ret


def train_epoch(epoch, model, data_loader, optimizer, criterion, logger):
    model.train()
    running_loss = 0.0
    prev_loss = 0.0
    n = 0
    log_interval = 100
    for batch_idx, (idx, seqs, emb, orig_lens) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(seqs)
        loss = criterion(output, emb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * seqs.size(0)
        n += seqs.size(0)
        if batch_idx % log_interval == 0:
            avg_loss = (running_loss - prev_loss) / n
            logger.info('[{:2d}, {:5d}] loss: {:.6f}'.format(epoch, batch_idx, avg_loss))
            prev_loss = running_loss
            n = 0
    return running_loss / len(data_loader.dataset)


def test_epoch(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for idx, seqs, emb, orig_lens in data_loader:
            output = model(seqs)
            running_loss += criterion(output, emb).item() * seqs.size(0)
    return running_loss / len(data_loader.dataset)


@docval({'name': 'input', 'type': str,
         'help': 'the input dataset'},

        {'name': 'model', 'type': nn.Module,
         'help': 'the model to train'},

        {'name': 'optimizer', 'type': optim.Optimizer,
         'help': 'the optimizer to use'},

        {'name': 'split_seed', 'type': int, 'default': None,
         'help': 'the seed to use for train-test split'},

        {'name': 'gpu', 'type': bool, 'default': False,
         'help': 'use GPU'},

        {'name': 'cuda_index', 'type': (int, list), 'default': 0,
         'help': 'which CUDA device to use'},

        {'name': 'epochs', 'type': int, 'default': 1,
         'help': 'the number of epochs to run'},

        {'name': 'batch_size', 'type': int, 'default': 64,
         'help': 'the batch size to use'},

        {'name': 'train_size', 'type': (int, float), 'default': 0.8,
         'help': 'the size of the training split'},

        {'name': 'output', 'type': str, 'default': None,
         'help': 'the file to save output (i.e. checkpoint) to'},

        {'name': 'checkpoint', 'type': (str, bool), 'default': False,
         'help': 'If True, resume from checkpoint in output. If string, assume string is path to checkpoint file'},

        {'name': 'load', 'type': bool, 'default': False,
         'help': 'load data into memory before training loop'},

        {'name': 'ohe', 'type': bool, 'default': True,
         'help': 'One-hot encode sequence data (only applies to protein data)'},

        {'name': 'logger', 'type': logging.Logger, 'default': None,
         'help': 'the path to the log file to use'},

        {'name': 'debug', 'type': bool, 'default': False,
         'help': 'run in debug mode (one batch per epoch)'},

        {'name': 'downsample', 'type': float, 'default': None,
         'help': 'downsample *input* before running'},

        {'name': 'pad', 'type': bool, 'default': False,
         'help': 'pad the sequences to be the maximum sequence length'},

        {'name': 'prof', 'type': str, 'default': None,
         'help': 'profile training loop and dump results to given path'},

        {'name': 'sanity', 'type': bool, 'default': False,
         'help': 'sanity check by copying response data into inputs'},

        {'name': '', 'type': None, 'help': '', 'default': None},
        is_method=False, allow_extra=True)
def run_serial(**kwargs):
    """
    Run training on a single process
    """

    model = kwargs['model']
    input = kwargs['input']
    optimizer = kwargs['optimizer']
    gpu = kwargs['gpu']
    epochs = kwargs['epochs']
    checkpoint = kwargs['checkpoint']
    output = kwargs['output']
    split_seed = kwargs['split_seed']
    batch_size = kwargs['batch_size']
    train_size = kwargs['train_size']
    cuda_index = kwargs['cuda_index']
    load = kwargs['load']

    logger = kwargs['logger']
    debug = kwargs['debug']
    prof = kwargs['prof']
    ohe = kwargs['ohe']
    pad = kwargs['pad']
    sanity = kwargs['sanity']
    downsample = kwargs['downsample']


    # set up logger
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    if debug:
        logger.setLevel(logging.DEBUG)

    device = None
    if gpu:
        to_index = cuda_index
        if isinstance(cuda_index, list):
            to_index = cuda_index[0]
            model = nn.DataParallel(model, device_ids=cuda_index)
            logger.info(f'running on GPUs {cuda_index}')
        device = torch.device(f"cuda:{to_index}")
        logger.info(f'sending data to CUDA device {str(device)}')
        model.to(device)

    if isinstance(checkpoint, bool):
        if checkpoint:
            checkpoint = output

    curr_epoch = 0

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    best_epoch = 0         # the epoch of the best model state
    best_state = None      # the best model state i.e. state that achieves lowest test loss
    if checkpoint:
        logger.info('picking up from checkpoint %s' % checkpoint)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        curr_epoch = checkpoint['epoch'] + 1
        train_loss = np.append(checkpoint['train_loss'], np.zeros(epochs))
        test_loss = np.append(checkpoint['test_loss'], np.zeros(epochs))
        split_seed = checkpoint['split_seed']
        batch_size = checkpoint['batch_size']
        best_epoch = checkpoint['best_epoch']
        best_state = checkpoint['best_state']

    logger.info('Optimizer:')
    logger.info(str(optimizer).replace('\n', '\n' + (' '*25)))
    logger.info('Model:')
    logger.info(str(model).replace('\n', '\n' + (' '*25)))


    last_epoch = curr_epoch + epochs
    criterion = nn.MSELoss()

    if downsample is not None:
        logger.info(f'downsampling dataset by a factor of {downsample}')

    logger.info('loading data from %s' % input)
    logger.info(f'- using {split_seed} as seed for train-test split')
    logger.info(f'- batch size: {batch_size}')
    if sanity:
        logger.info('running sanity check. i.e. copying response data into inputs')
    train_loader, test_loader, validate_loader = train_test_loaders(input,
                                                                    batch_size=batch_size,
                                                                    device=device,
                                                                    load=load,
                                                                    ohe=ohe,
                                                                    pad=pad,
                                                                    sanity=sanity,
                                                                    downsample=downsample,
                                                                    random_state=split_seed)

    logger.info(f'- train size:    {len(train_loader.sampler.indices)}')
    logger.info(f'- test size:     {len(test_loader.sampler.indices)}')
    logger.info(f'- validate size: {len(validate_loader.sampler.indices)}')

    logger.info(f'starting with epoch {curr_epoch+1}')
    logger.info(f'saving results to {output}')

    nprint = (len(train_loader.dataset) / batch_size) // 10

    if prof:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    before = datetime.now()
    for curr_epoch in range(curr_epoch, last_epoch):  # loop over the dataset multiple times
        logger.info(f'begin epoch {curr_epoch+1}')


        train_loss[curr_epoch] = train_epoch(curr_epoch, model, train_loader, optimizer, criterion, logger)
        test_loss[curr_epoch] = test_epoch(model, test_loader, criterion)
#        model.train()
#        prev_loss = 0.0
#        for i, data in enumerate(train_loader, 0):
#            # get the inputs; data is a list of [inputs, labels]
#            idx, seqs, emb, orig_lens = data
#
#            # zero the parameter gradients
#            optimizer.zero_grad()
#
#            ## Modify here to adjust learning rate
#            ## Create new Optimizer object?
#
#            # forward + backward + optimize
#            logger.debug('forward')
#            outputs = model(seqs, orig_len=orig_lens)
#
#            logger.debug('criterion')
#            loss = criterion(outputs, emb)
#            logger.debug('backward')
#            loss.backward()
#            logger.debug('step')
#            optimizer.step()
#
#            train_loss[curr_epoch] += loss.item()
#            if i % nprint == nprint - 1:    # print every 10th of training set
#                logger.info('[%d, %5d] loss: %.6e' %
#                            (curr_epoch + 1, i + 1, (train_loss[curr_epoch]-prev_loss)/nprint))
#                prev_loss = train_loss[curr_epoch]
#            if debug:
#                if i == 1:
#                    break
#
#        model.eval()
#        logger.debug('test loss')
#        for i, data in enumerate(test_loader, 0):
#            idx, seqs, emb, orig_lens = data
#            outputs = model(seqs, orig_len=orig_lens)
#            crit = criterion(outputs, emb).item()
#            test_loss[curr_epoch] += crit
#            if debug:
#                break
#
        logger.info(f'epoch {curr_epoch+1} complete')
        logger.info(f'- training loss: {train_loss[curr_epoch]}; test loss: {test_loss[curr_epoch]}')

        if curr_epoch > 0 and test_loss[curr_epoch] <= test_loss[best_epoch]:
            logger.info(f'updating best state')
            logger.info(f'- previous best loss: {test_loss[best_epoch]} (epoch {best_epoch+1})')
            best_epoch = curr_epoch
            best_state = model.state_dict()
        else:
            # lower learning rate if we don't get any better for 5 epochs
            if curr_epoch - best_epoch > 5:
                pass
                # # disable this for now
                # logger.info('Reducing learning rate:')
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = param_group['lr'] * 0.1
                # logger.info(str(optimizer).replace('\n', '\n' + (' '*25)))

        logger.debug('checkpointing')
        torch.save({'epoch': curr_epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'best_epoch': best_epoch,
                    'best_state': best_state,
                    'split_seed': split_seed,
                    'batch_size': batch_size,
                    }, output)


    if prof:
        pr.disable()
        pr.dump_stats(prof)

    after = datetime.now()
    logger.info('Finished Training. Took %s seconds' % (after-before).total_seconds())


