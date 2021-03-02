import sys
import os
import os.path
import pickle
import json
from datetime import datetime
import numpy as np
from ..utils import parse_seed, check_argv, parse_logger, check_directory
from .utils import process_gpus, process_model, process_output
from .loader import add_dataset_arguments
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
    parser.add_argument('model', type=str, help='the model to run. see show-models for a list of available models',
                        metavar="model", choices=list(models._models.keys()))
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('output', type=str, help='file to save model', default=None)

    add_dataset_arguments(parser)
    parser.add_argument('--add_clf', default=False, action='store_true', help='add a classifier to a pretrained model')

    parser.add_argument('-l', '--load', action='store_true', default=False, help='load data into memory before running training loop')

    parser.add_argument('-o', '--n_outputs', type=int, help='the number of outputs in the final layer', default=32)
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-T', '--test', action='store_true', help='run test data through model', default=False)
    parser.add_argument('-A', '--accumulate', type=json.loads, help='accumulate_grad_batches argument to pl.Trainer', default=1)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
    parser.add_argument('-D', '--dropout_rate', type=float, help='the dropout rate to use', default=0.5)
    parser.add_argument('-p', '--protein', action='store_true', default=False, help='input contains protein sequences')
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-n', '--num_nodes', type=int, default=1, help='the number of nodes to run on')
    parser.add_argument('-s', '--seed', type=parse_seed, default='', help='seed to use for train-test split')
    parser.add_argument('-t', '--train_size', type=parse_train_size, default=0.8, help='size of train split')
    parser.add_argument('-H', '--hparams', type=json.loads, help='additional hparams for the model. this should be a JSON string', default=None)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('--fp16', action='store_true', default=False, help='use 16-bit training')
    parser.add_argument('-E', '--experiment', type=str, default='default', help='the experiment name')
    parser.add_argument('--profile', action='store_true', default=False, help='profile with PyTorch Lightning profile')
    parser.add_argument('--sanity', action='store_true', default=False, help='copy response data into input data')
    parser.add_argument('-r', '--lr', type=float, default=0.01, help='the learning rate for Adam')
    parser.add_argument('--lr_find', default=False, action='store_true', help='find optimal learning rate')
    parser.add_argument('--lr_scheduler', default='adam', choices=AbstractLit.schedules, help='the learning rate schedule to use')
    grp = parser.add_argument_group('Distributed training environments').add_mutually_exclusive_group()
    grp.add_argument('--horovod', default=False, action='store_true', help='run using Horovod backend')
    grp.add_argument('--lsf', default=False, action='store_true', help='running on LSF system')
    grp.add_argument('--slurm', default=False, action='store_true', help='running on SLURM system')


    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    return args

def which(program):
    """
    Use to check for resource managers
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def process_args(args=None, return_io=False):
    """
    Process arguments for running training
    """
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args)

    # determing number of input channels:
    # 18 for DNA, 26 for protein
    # 5 for sanity check (this probably doesn't work anymore)
    input_nc = 18
    if args.protein:
        input_nc = 26
    if args.sanity:
        input_nc = 5
    args.input_nc = input_nc

    args.loader_kwargs = dict()
    if args.lsf:
        args.loader_kwargs['num_workers'] = 6
        args.loader_kwargs['multiprocessing_context'] = 'spawn'

    model = process_model(args)

    targs = dict(
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
    )

    if args.profile:
        targs['profiler'] = True

    targs['accumulate_grad_batches'] = args.accumulate

    #if args.gpus is not None:
    #    targs['gpus'] = 1
    #targs['accelerator'] = 'horovod'

    if args.horovod:
        targs['accelerator'] = 'horovod'
        if args.gpus:
            targs['gpus'] = 1
    else:
        targs['gpus'] = process_gpus(args.gpus)
        targs['num_nodes'] = args.num_nodes
        targs['accelerator'] = 'ddp'
        if targs['gpus'] != 1 or targs['num_nodes'] > 1:
            # env = None
            # if args.lsf:
            #     env = cenv.LSFEnvironment()
            # elif args.slurm:
            #     env = cenv.SLURMEnvironment()
            # else:
            #     print("If running multi-node or multi-gpu, you must specify resource manager, i.e. --lsf or --slurm",
            #           file=sys.stderr)
            #     sys.exit(1)
            # targs.setdefault('plugins', list()).append(env)
            pass
    del args.gpus

    if args.debug:
        targs['fast_dev_run'] = True

    if args.lr_find:
        targs['auto_lr_find'] = True
    del args.lr_find

    if args.fp16:
        targs['amp_level'] = 'O2'
        targs['precision'] = 16
    del args.fp16

    ret = [model, args, targs]
    if return_io:
        ret.append(io)

    if args.checkpoint:
        args.experiment += '_restart'

    return tuple(ret)


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def print0(*msg, **kwargs):
    rank = 0
    if rank == 0:
        print(*msg, **kwargs)


def run_lightning(argv=None):
    '''Run training with PyTorch Lightning'''

    print0(argv)
    model, args, addl_targs = process_args(parse_args(argv=argv))

    outbase, output = process_output(args)
    check_directory(outbase)
    print0(args)

    # save arguments
    with open(output('args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    seed_everything(args.seed)

    # get dataset so we can set model parameters that are
    # dependent on the dataset, such as final number of outputs

    targs = dict(
        checkpoint_callback=ModelCheckpoint(dirpath=outbase, save_weights_only=False, save_last=True, save_top_k=1, monitor=AbstractLit.val_loss),
        logger = CSVLogger(save_dir=os.path.join(args.output, 'logs')),
    )
    targs.update(addl_targs)

    if args.debug:
        targs['log_every_n_steps'] = 1

    print0('Trainer args:', targs, file=sys.stderr)
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

    print0("Took %02d:%02d:%02d.%d" % (hours,minutes,seconds,td.microseconds), file=sys.stderr)
    print0("Total seconds:", td.total_seconds(), file=sys.stderr)


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

def cuda_sum(argv=None):
    '''Summarize what Torch sees in CUDA land'''
    import torch
    print('torch.cuda.is_available:', torch.cuda.is_available())
    print('torch.cuda.device_count:', torch.cuda.device_count())

def print_dataloader(dl):
    print(dl.dataset.index[0], dl.dataset.index[-1])

def overall_metric(model, loader, metric):
    val = 0.0
    for idx, seqs, target, olen in loader:
        output = model(seqs)
        val += metric(target, output)
    return val

def print_args(argv=None):
    '''Run Lightning Learning Rate finder'''
    import pickle
    import ruamel.yaml as yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', type=str, help='the args.pkl file')
    args = parser.parse_args(argv)
    with open(args.pkl, 'rb') as f:
        namespace = pickle.load(f)
    yaml.main.safe_dump(vars(namespace), sys.stdout, default_flow_style=False)




from . import models
from .models.lit import AbstractLit

if __name__ == '__main__':
    run_lightening()
