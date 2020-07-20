import sys
import os
import os.path
import pickle
from datetime import datetime
import numpy as np
from ..utils import parse_seed, check_argv, parse_logger
from .utils import process_gpus, process_model, process_output
from hdmf.utils import docval

import argparse
import logging


def parse_train_size(string):
    ret = float(string)
    if ret > 1.0:
        ret = int(ret)
    return ret


def parse_args(*addl_args, argv=None):
    """
    Parse arguments for training executable
    """
    import json
    argv = check_argv(argv)

    epi = """
    output can be used as a checkpoint
    """
    desc = "Run network training"
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('model', type=str, help='the model to run', choices=list(models._models.keys()))
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('output', type=str, help='file to save model', default=None)
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument('-C', '--classify', action='store_true', help='run a classification problem', default=False)
    type_group.add_argument('-M', '--manifold', action='store_true', help='run a manifold learning problem', default=False)
    type_group.add_argument('-R', '--regression', action='store_true', help='run a regression problem', default=False)
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-T', '--test', action='store_true', help='run test data through model', default=False)
    parser.add_argument('-A', '--accuracy', action='store_true', help='compute accuracy', default=False)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
    parser.add_argument('-D', '--dropout_rate', type=float, help='the dropout rate to use', default=0.5)
    parser.add_argument('-p', '--protein', action='store_true', default=False, help='input contains protein sequences')
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-s', '--seed', type=parse_seed, default='', help='seed to use for train-test split')
    parser.add_argument('-t', '--train_size', type=parse_train_size, default=0.8, help='size of train split')
    parser.add_argument('-H', '--hparams', type=json.loads, help='additional hparams for the model. this should be a JSON string', default=None)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('--downsample', type=float, default=None, help='downsample input before training')
    parser.add_argument('-E', '--experiment', type=str, default='default', help='the experiment name')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('--prof', type=str, default=None, metavar='PATH', help='profile training loop dump results to PATH')
    parser.add_argument('--sanity', action='store_true', default=False, help='copy response data into input data')
    parser.add_argument('-L', '--load', action='store_true', default=False, help='load data into memory before running training loop')
    parser.add_argument('--lr', type=float, default=0.01, help='the learning rate for Adam')
    parser.add_argument('--lr_find', default=False, action='store_true', help='find optimal learning rate')
    parser.add_argument('-W', '--window', type=int, default=None, help='the window size to use to chunk sequences')
    parser.add_argument('-S', '--step', type=int, default=None, help='the step between windows. default is to use window size (i.e. non-overlapping chunks)')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    return args


def process_args(args=None, return_io=False):
    """
    Process arguments for running training
    """
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args)

    logger = args.logger
    # set up logger
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    args.logger = logger

    # determing number of input channels:
    # 5 for DNA, 26 for protein
    # 5 for sanity check (this probably doesn't work anymore)
    input_nc = 5
    if args.protein:
        input_nc = 26
    if args.sanity:
        input_nc = 5
    args.input_nc = input_nc

    model = process_model(args)

    targs = dict(
        max_epochs=args.epochs,
    )

    targs['gpus'] = process_gpus(args.gpus)
    if targs['gpus'] != 1:
        targs['distributed_backend'] = 'ddp'
    del args.gpus

    if args.debug:
        targs['fast_dev_run'] = True

    if args.lr_find:
        targs['auto_lr_find'] = True
    del args.lr_find

    ret = [model, args, targs]
    if return_io:
        ret.append(io)

    return tuple(ret)


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .. import command
@command('train')
def run_lightning(argv=None):
    '''Run training with PyTorch Lightning'''
    model, args, addl_targs = process_args(parse_args(argv=argv))

    outbase, output = process_output(args)
    print(args)
    del args.logger 
    
    # save arguments
    with open(output('args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    seed_everything(args.seed)

    # get dataset so we can set model parameters that are
    # dependent on the dataset, such as final number of outputs

    targs = dict(
        checkpoint_callback=ModelCheckpoint(filepath=output('seed=%d-{epoch:02d}-{val_loss:.2f}' % args.seed), save_weights_only=False),
        logger = TensorBoardLogger(save_dir=os.path.join(args.output, 'tb_logs'), name=args.experiment),
        row_log_interval=10,
        log_save_interval=100
    )
    targs.update(addl_targs)

    trainer = Trainer(**targs)

    if args.debug:
        print_dataloader(model.test_dataloader())
        print_dataloader(model.train_dataloader())
        print_dataloader(model.val_dataloader())

    s = datetime.now()
    trainer.fit(model)
    e = datetime.now()
    td = e - s
    hours, seconds = divmod(td.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    print("Took %02d:%02d:%02d.%d" % (hours,minutes,seconds,td.microseconds))

@command('lr-find')
def lightning_lr_find(argv=None):
    '''Run Lightning Learning Rate finder'''
    import matplotlib.pyplot as plt

    model, args, addl_targs = process_args(parse_args(argv=argv))

    outbase, output = process_output(args, subdir='lr_find')

    # save arguments
    with open(output('args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    seed_everything(args.seed)

    # get dataset so we can set model parameters that are
    # dependent on the dataset, such as final number of outputs

    targs = addl_targs

    trainer = Trainer(**targs)

    s = datetime.now()
    lr_finder = trainer.lr_find(model)
    e = datetime.now()
    td = e - s
    hours, seconds = divmod(td.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    print("Took %02d:%02d:%02d.%d" % (hours,minutes,seconds,td.microseconds))
    print('optimal learning rate: %0.6f' % lr_finder.suggestion())
    fig = lr_finder.plot(suggest=True)
    fig.savefig(output('lr_finder_results.png'))

def print_dataloader(dl):
    print(dl.dataset.index[0], dl.dataset.index[-1])

def overall_metric(model, loader, metric):
    val = 0.0
    for idx, seqs, target, olen in loader:
        output = model(seqs)
        val += metric(target, output)
    return val


from . import models

if __name__ == '__main__':
    run_lightening()
