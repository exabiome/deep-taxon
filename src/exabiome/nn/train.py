import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from exabiome.sequence import AbstractChunkedDIFile, WindowChunkedDIFile
from . import SeqDataset, train_test_loaders
from ..utils import parse_seed
from hdmf.utils import docval
from hdmf.common import get_hdf5io

import argparse
import logging


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
    parser.add_argument('-C', '--classify', action='store_true', help='run a classification problem', default=False)
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-r', '--resume', action='store_true', help='resume training from checkpoint stored in output', default=False)
    parser.add_argument('-V', '--validate', action='store_true', help='run validation data through model', default=False)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
    parser.add_argument('-p', '--protein', action='store_true', default=False, help='input contains protein sequences')
    parser.add_argument('-g', '--gpu', action='store_true', default=False, help='use GPU')
    parser.add_argument('-i', '--cuda_index', type=parse_cuda_index, default='all', help='which CUDA device to use')
    parser.add_argument('-s', '--split_seed', type=parse_seed, default='', help='seed to use for train-test split')
    parser.add_argument('-t', '--train_size', type=parse_train_size, default=0.8, help='size of train split')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('-D', '--downsample', type=float, default=None, help='downsample input before training')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('--prof', type=str, default=None, metavar='PATH', help='profile training loop dump results to PATH')
    parser.add_argument('--sanity', action='store_true', default=False, help='copy response data into input data')
    parser.add_argument('-L', '--load', action='store_true', default=False, help='load data into memory before running training loop')
    parser.add_argument('--lr', type=float, default=0.01, help='the learning rate for Adam')
    parser.add_argument('-W', '--window', type=int, default=None, help='the window size to use to chunk sequences')
    parser.add_argument('-S', '--step', type=int, default=None, help='the step between windows. default is to use window size (i.e. non-overlapping chunks)')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    ret = vars(args)

    logger = ret['logger']
    # set up logger
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    ret['logger'] = logger

    if args.checkpoint is None:
        ret['checkpoint'] = False
    if args.resume:
        if not isinstance(ret['checkpoint'], str):
            # don't overwrite a given path
            ret['checkpoint'] = True
    ret.pop('resume')

    if ret.pop('gpu'):
        cuda_index = ret['cuda_index']
        to_index = cuda_index
        if isinstance(cuda_index, list):
            to_index = cuda_index[0]
        ret['device'] = torch.device(f"cuda:{to_index}")


    return ret


def check_window(window, step):
    if window is None:
        return None, None
    else:
        if step is None:
            step = window
        return window, step


def load_dataset(path, load=False, ohe=True, device=None, pad=False, sanity=False,
                 protein=False, window=None, step=None, classify=False, **kwargs):
    hdmfio = get_hdf5io(path, 'r')
    difile = hdmfio.read()
    if load:
        difile.load()

    window, step = check_window(window, step)
    if window is not None:
        difile = WindowChunkedDIFile(difile, window, step)

    dataset = SeqDataset(difile, device=device, ohe=ohe, pad=pad, sanity=sanity, classify=classify)
    return dataset, hdmfio


def check_model(model, logger=None, device=None, cuda_index=0, **kwargs):
    if device is not None:
        to_index = cuda_index
        if isinstance(cuda_index, list):
            model = nn.DataParallel(model, device_ids=cuda_index)
            if logger:
                logger.info(f'running on GPUs {cuda_index}')
        if logger:
            logger.info(f'sending data to CUDA device {str(device)}')
        model.to(device)
    elif isinstance(cuda_index, list):
        model = nn.DataParallel(model)
    return model


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
    return running_loss / len(data_loader.sampler)


def test_epoch(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for idx, seqs, emb, orig_lens in data_loader:
            try:
                output = model(seqs)
            except Exception as e:
                print(idx)
                print(orig_lens)
                raise e
            running_loss += criterion(output, emb).item() * seqs.size(0)
    return running_loss / len(data_loader.sampler)


def validate_epoch(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0

    outputs = list()
    with torch.no_grad():
        for idx, seqs, emb, orig_lens in data_loader:
            try:
                outputs.append(model(seqs))
            except Exception as e:
                print(idx)
                print(orig_lens)
                raise e
            running_loss += criterion(outputs[-1], emb).item() * seqs.size(0)
    outputs = torch.cat(outputs)
    return running_loss / len(data_loader.sampler), outputs


@docval({'name': 'dataset', 'type': (SeqDataset, AbstractChunkedDIFile),
         'help': 'the input dataset'},

        {'name': 'model', 'type': nn.Module,
         'help': 'the model to train'},

        {'name': 'optimizer', 'type': optim.Optimizer,
         'help': 'the optimizer to use'},

        {'name': 'criterion', 'type': nn.Module, 'default': None,
         'help': 'the loss function to use'},

        {'name': 'split_seed', 'type': int, 'default': None,
         'help': 'the seed to use for train-test split'},

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

        {'name': 'downsample', 'type': float, 'default': None,
         'help': 'downsample *dataset* before running'},

        {'name': 'prof', 'type': str, 'default': None,
         'help': 'profile training loop and dump results to given path'},

        {'name': 'sanity', 'type': bool, 'default': False,
         'help': 'sanity check by copying response data into inputs'},

        {'name': 'classify', 'type': bool, 'default': False,
         'help': 'run a classification problem'},

        {'name': '', 'type': None, 'help': '', 'default': None},
        is_method=False, allow_extra=True)
def train_serial(**kwargs):
    """
    Run training on a single process
    """

    model = kwargs['model']
    dataset = kwargs['dataset']
    optimizer = kwargs['optimizer']
    epochs = kwargs['epochs']
    checkpoint = kwargs['checkpoint']
    output = kwargs['output']
    split_seed = kwargs['split_seed']
    batch_size = kwargs['batch_size']
    train_size = kwargs['train_size']
    load = kwargs['load']

    logger = kwargs['logger']
    prof = kwargs['prof']
    ohe = kwargs['ohe']
    sanity = kwargs['sanity']
    downsample = kwargs['downsample']

    criterion = kwargs['criterion'] or (nn.CrossEntropyLoss() if kwargs['classify'] else nn.MSELoss())

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

    if downsample is not None:
        logger.info(f'downsampling dataset by a factor of {downsample}')

    logger.info('loading data from %s' % input)
    logger.info(f'- using {split_seed} as seed for train-test split')
    logger.info(f'- batch size: {batch_size}')
    if sanity:
        logger.info('running sanity check. i.e. copying response data into inputs')
    train_loader, test_loader, validate_loader = train_test_loaders(dataset,
                                                                    batch_size=batch_size,
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


@docval({'name': 'checkpoint', 'type': str,
         'help': 'the path to the checkpoint file'},

        {'name': 'dataset', 'type': (SeqDataset, AbstractChunkedDIFile), 'default': None,
         'help': 'the input dataset'},

        {'name': 'model', 'type': nn.Module, 'default': None,
         'help': 'the model to load best state into'},

        {'name': 'current_state', 'type': bool, 'default': False,
         'help': 'load current state into model. load best state by default'},

        {'name': 'downsample', 'type': float, 'default': None,
         'help': 'downsample *dataset* before running'},

        is_method=False, allow_extra=True)
def load_checkpoint(**kwargs):
    checkpoint = kwargs['checkpoint']
    dataset = kwargs['dataset']
    model = kwargs['model']
    current_state = kwargs['current_state']
    downsample = kwargs['downsample']

    map_location=None
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    checkpoint = torch.load(checkpoint, map_location=map_location)
    ret = checkpoint
    downsample = checkpoint.get('downsample', downsample)

    if model is not None:
        if current_state:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['best_state'])

    if dataset is not None:
        train, test, validate = train_test_loaders(dataset,
                                                   batch_size=checkpoint['batch_size'],
                                                   downsample=downsample,
                                                   random_state=checkpoint['split_seed'])
        ret['train'] = train
        ret['test'] = test
        ret['validate'] = validate


    ret['n_epochs'] = checkpoint['epoch']
    ret['model'] = model

    return ret


def run(dataset, model, **args):

    if args['validate']:
        output = args.pop('output')
        model = check_model(model, **args)
        cp = load_checkpoint(output, model=model, dataset=dataset, downsample=args.get('downsample'))
        loader = cp['validate']
        criterion = args.get('criterion', nn.MSELoss())
        loss, outputs = validate_epoch(model, loader, criterion)
        return loss, outputs, cp
    else:
        optimizer = args.get('optimizer')
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=args['lr'])
        train_serial(dataset=dataset, model=model, optimizer=optimizer, **args)



