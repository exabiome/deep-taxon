import sys
import os
import os.path
import pickle
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

from . import models

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


def parse_model(string):
    return models[string]


def parse_args(*addl_args, argv=None):
    """
    Parse arguments for training executable
    """
    if argv is None:
        argv = sys.argv[1:]

    epi = """
    output can be used as a checkpoint
    """
    desc = "Run network training"
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('model', type=str, help='the model to run', choices=list(models._models.keys()))
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('output', type=str, help='file to save model', default=None)
    parser.add_argument('-C', '--classify', action='store_true', help='run a classification problem', default=False)
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-r', '--resume', action='store_true', help='resume training from checkpoint stored in output', default=False)
    parser.add_argument('-T', '--test', action='store_true', help='run test data through model', default=False)
    parser.add_argument('-A', '--accuracy', action='store_true', help='compute accuracy', default=False)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
    parser.add_argument('-p', '--protein', action='store_true', default=False, help='input contains protein sequences')
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    #parser.add_argument('-i', '--cuda_index', type=parse_cuda_index, default='all', help='which CUDA device to use')
    parser.add_argument('-s', '--seed', type=parse_seed, default='', help='seed to use for train-test split')
    parser.add_argument('-t', '--train_size', type=parse_train_size, default=0.8, help='size of train split')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('-D', '--downsample', type=float, default=None, help='downsample input before training')
    parser.add_argument('-E', '--experiment', type=str, default='default', help='the experiment name')
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

    logger = args.logger
    # set up logger
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    args.logger = logger

    model = models._models[args.model]
    input_path = args.input

    # determing number of input channels:
    # 5 for DNA, 26 for protein
    # 5 for sanity check (this probably doesn't work anymore)
    input_nc = 5
    if args.protein:
        input_nc = 26
    if args.sanity:
        input_nc = 5
    args.input_nc = input_nc

    args.window, args.step = check_window(args.window, args.step)

    del args.resume
    del args.input
    del args.model

    return model, input_path, args


def check_window(window, step):
    if window is None:
        return None, None
    else:
        if step is None:
            step = window
        return window, step


def get_dataset(path, protein=False, window=None, step=None, classify=False, **kwargs):
    hdmfio = get_hdf5io(path, 'r')
    difile = hdmfio.read()
    dataset = SeqDataset(difile, classify=classify)
    return dataset, hdmfio


def _check_dir(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError(f'{path} already exists as a file')
    else:
        os.makedirs(path)


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def run_lightening():
    lit_cls, input_path, args = parse_args()

    outbase = args.output
    if args.experiment:
        outbase = os.path.join(outbase, 'training_results', args.experiment)
    _check_dir(outbase)

    def output(fname):
        return os.path.join(outbase, fname)

    # save arguments
    with open(output('args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    seed_everything(args.seed)

    dataset, io = get_dataset(input_path,
                              protein=args.protein,
                              window=args.window,
                              step=args.step,
                              classify=args.classify)
    dataset.load()

    if args.classify:
        n_outputs = len(dataset.difile.taxa_table)
    else:
        n_outputs = dataset.difile.n_emb_components
    args.n_outputs = n_outputs

    targs = dict(
        max_epochs=args.epochs,
        checkpoint_callback=ModelCheckpoint(filepath=output('seed=%d-{epoch:02d}-{val_loss:.2f}' % args.seed), save_weights_only=False),
        logger = TensorBoardLogger(save_dir=os.path.join(args.output, 'tb_logs'), name=args.experiment),
        row_log_interval=10,
        log_save_interval=100
    )

    gpus = args.gpus
    if isinstance(gpus, str):
        gpus = [int(g) for g in gpus.split(',')]
        targs['distributed_backend'] = 'dp'
    else:
        gpus = 1 if gpus else None
    targs['gpus'] = gpus

    if args.debug:
        targs['fast_dev_run'] = True

    trainer = Trainer(**targs)
    if args.test:
        if args.checkpoint is not None:
            net = lit_cls.load_from_checkpoint(args.checkpoint)
        else:
            print('If running with --test, must provide argument to --checkpoint', file=sys.stderr)
            sys.exit(1)

        net.set_dataset(dataset)

        print_dataloader(net.test_dataloader())
        print_dataloader(net.train_dataloader())
        print_dataloader(net.val_dataloader())
        if args.accuracy:
            from .metric import NCorrect, NeighborNCorrect
            if args.classify:
                metric = NCorrect()
            else:
                metric = NeighborNCorrect(dataset.difile)
            total_correct = overall_metric(net, net.test_dataloader(), metric)
            print(total_correct/len(net.test_dataloader().sampler))
        else:
            trainer.test(net)
    else:
        if args.checkpoint is not None:
            net = lit_cls.load_from_checkpoint(args.checkpoint)
        else:
            net = lit_cls(args)

        net.set_dataset(dataset)

        print_dataloader(net.test_dataloader())
        print_dataloader(net.train_dataloader())
        print_dataloader(net.val_dataloader())
        trainer.fit(net)

def print_dataloader(dl):
    print(dl.sampler.indices[0], dl.sampler.indices[-1])

def overall_metric(model, loader, metric):
    val = 0.0
    for idx, seqs, target, olen in loader:
        output = model(seqs)
        val += metric(target, output)
    return val


from . import models

if __name__ == '__main__':
    run_lightening()
