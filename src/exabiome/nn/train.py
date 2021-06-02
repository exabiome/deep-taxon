import sys
import os
import os.path
import pickle
import json
from datetime import datetime
import numpy as np
from ..utils import parse_seed, check_argv, parse_logger, check_directory
from .utils import process_gpus, process_model, process_output
from .loader import add_dataset_arguments, DeepIndexDataModule
from hdmf.utils import docval

import argparse
import logging


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.accelerators import GPUAccelerator, CPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin, SingleDevicePlugin
from .lsf_environment import LSFEnvironment

import torch

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

    ##################################################
    # Add dataset-specific arguments, like tgt_tax_lvl
    ##################################################
    add_dataset_arguments(parser)

    parser.add_argument('-F', '--features_checkpoint', type=str, help='a checkpoint file for previously trained features', default=None)

    parser.add_argument('-l', '--load', action='store_true', default=False, help='load data into memory before running training loop')

    parser.add_argument('-w', '--weighted', nargs='?', const=True, default=False, choices=['ins', 'isns', 'ens'], help='weight classes in classification')
    parser.add_argument('--ens_beta', type=float, help='the value of beta to use when weighting with effective number of sample (ens)', default=0.9)
    parser.add_argument('-o', '--n_outputs', type=int, help='the number of outputs in the final layer', default=32)
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-T', '--test', action='store_true', help='run test data through model', default=False)
    parser.add_argument('-A', '--accumulate', type=json.loads, help='accumulate_grad_batches argument to pl.Trainer', default=1)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
    parser.add_argument('-D', '--dropout_rate', type=float, help='the dropout rate to use', default=0.5)
    parser.add_argument('-p', '--protein', action='store_true', default=False, help='input contains protein sequences')
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-n', '--num_nodes', type=int, default=1, help='the number of nodes to run on')
    parser.add_argument('-s', '--seed', type=parse_seed, default='', help='seed to use for train-test split')
    parser.add_argument('-H', '--hparams', type=json.loads, help='additional hparams for the model. this should be a JSON string', default=None)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('--fp16', action='store_true', default=False, help='use 16-bit training')
    parser.add_argument('-E', '--experiment', type=str, default='default', help='the experiment name')
    parser.add_argument('--profile', action='store_true', default=False, help='profile with PyTorch Lightning profile')
    parser.add_argument('--sanity', action='store_true', default=False, help='copy response data into input data')
    parser.add_argument('-r', '--lr', type=float, default=0.001, help='the learning rate for Adam')
    parser.add_argument('-O', '--optimizer', type=str, choices=['adam', 'lamb'], help='the optimizer to use', default='adam')
    parser.add_argument('--lr_find', default=False, action='store_true', help='find optimal learning rate')
    parser.add_argument('--lr_scheduler', default='adam', choices=AbstractLit.schedules, help='the learning rate schedule to use')
    grp = parser.add_argument_group('Distributed training environments').add_mutually_exclusive_group()
    grp.add_argument('--horovod', default=False, action='store_true', help='run using Horovod backend')
    grp.add_argument('--lsf', default=False, action='store_true', help='running on LSF system')
    grp.add_argument('--slurm', default=False, action='store_true', help='running on SLURM system')

    dl_grp = parser.add_argument_group('Data loading')
    dl_grp.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
    dl_grp.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=1)
    dl_grp.add_argument('-y', '--pin_memory', action='store_true', default=False, help='pin memory when loading data')
    dl_grp.add_argument('-f', '--shuffle', action='store_true', default=False, help='shuffle batches when training')


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

    # classify by default
    if args.manifold == args.classify == False:
        args.classify = True

    # make sure we are classifying if we are using adding classifier layers
    # to a resnet features model
    if args.features_checkpoint is not None:
        if args.manifold:
            raise ValueError('Cannot use manifold loss (i.e. -M) if adding classifier (i.e. -F)')
        args.classify = True

    data_mod = DeepIndexDataModule(args)

    #model = process_model(args, taxa_table=data_mod.dataset.difile.taxa_table)

    # n_taxa replaces n_outputs
    args.n_taxa = len(data_mod.dataset.taxa_labels)
    model = process_model(args)

    if args.weighted:
        #labels = data_mod.dataset.difile.taxa_table[args.tgt_tax_lvl].data
        if args.weighted == 'ens':
            weights = (1 - args.ens_beta)/(1 - args.ens_beta**data_mod.dataset.taxa_counts)
        elif args.weighted == 'isns':
            weights = np.sqrt(1/data_mod.dataset.taxa_counts)
        else:
            weights = np.sqrt(1/data_mod.dataset.taxa_counts)
        model.set_class_weights(weights)

    targs = dict(
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
    )

    if args.profile:
        targs['profiler'] = True

    targs['accumulate_grad_batches'] = args.accumulate

    targs['gpus'] = process_gpus(args.gpus)

    #if args.horovod:
    #    targs['accelerator'] = 'horovod'
    #    if args.gpus:
    #        targs['gpus'] = 1
    #else:
    #    targs['num_nodes'] = args.num_nodes
    #    targs['accelerator'] = 'ddp'
    #    if targs['gpus'] != 1 or targs['num_nodes'] > 1:
    #        # env = None
    #        # if args.lsf:
    #        #     env = cenv.LSFEnvironment()
    #        # elif args.slurm:
    #        #     env = cenv.SLURMEnvironment()
    #        # else:
    #        #     print("If running multi-node or multi-gpu, you must specify resource manager, i.e. --lsf or --slurm",
    #        #           file=sys.stderr)
    #        #     sys.exit(1)
    #        # targs.setdefault('plugins', list()).append(env)
    #        pass
    del args.gpus


    env = None
    if args.lsf:
        ##########################################################################################
        # Currently coding against pytorch-lightning 1.3.1.
        ##########################################################################################
        args.loader_kwargs['num_workers'] = 1
        args.loader_kwargs['multiprocessing_context'] = 'spawn'
        env = LSFEnvironment()
    elif args.slurm:
        env = SLURMEnvironment()

    if targs['gpus'] is not None:
        if targs['gpus'] == 1:
            targs['accelerator'] = GPUAccelerator(
                precision_plugin = NativeMixedPrecisionPlugin(),
                training_type_plugin = SingleDevicePlugin(torch.device(0))
            )
        else:
            if env is None:
                raise ValueError('Please specify environment (--lsf or --slurm) if using more than one GPU')
            parallel_devices = [torch.device(i) for i in range(torch.cuda.device_count())]
            targs['accelerator'] = GPUAccelerator(
                precision_plugin = NativeMixedPrecisionPlugin(),
                training_type_plugin = DDPPlugin(parallel_devices=parallel_devices,
                                                 cluster_environment=env)
            )
    else:
        targs['accelerator'] = CPUAccelerator(
            training_type_plugin = DDPPlugin(cluster_environment=env, num_nodes=args.num_nodes)
        )

    if args.debug:
        targs['limit_train_batches'] = 20
        targs['limit_val_batches'] = 5
        targs['max_epochs'] = 5

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

    ret.append(data_mod)

    return tuple(ret)


#from mpi4py import MPI
#
#comm = MPI.COMM_WORLD
#RANK = comm.Get_rank()
RANK = 0

def print0(*msg, **kwargs):
    if RANK == 0:
        print(*msg, **kwargs)

def run_lightning(argv=None):
    '''Run training with PyTorch Lightning'''

    import signal
    import traceback
    def signal_handler(sig, frame):
        print('I caught SIG_KILL!')
        track = traceback.format_exc()
        print(track)
        raise KeyboardInterrupt
        sys.exit(0)
    signal.signal(signal.SIGTERM, signal_handler)

    print0(argv)
    model, args, addl_targs, data_mod = process_args(parse_args(argv=argv))

    # output is a wrapper function for os.path.join(outdir, <FILE>)
    outdir, output = process_output(args)
    check_directory(outdir)
    print0(args)

    # save arguments
    with open(output('args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    if args.checkpoint is not None:
        if RANK == 0:
            print0(f'symlinking to {args.checkpoint} from {outdir}')
            dest = output('start.ckpt')
            src = os.path.relpath(args.checkpoint, start=outdir)
            if os.path.exists(dest):
                existing_src = os.readlink(dest)
                if existing_src != src:
                    msg = f'Cannot create symlink to checkpoint -- {dest} already exists, but points to {existing_src}'
                    raise RuntimeError(msg)
            else:
                os.symlink(src, dest)

    seed_everything(args.seed)

    # get dataset so we can set model parameters that are
    # dependent on the dataset, such as final number of outputs

    targs = dict(
        checkpoint_callback=ModelCheckpoint(dirpath=outdir, save_weights_only=False, save_last=True, save_top_k=1, monitor=AbstractLit.val_loss),
        logger = CSVLogger(save_dir=output('logs')),
        profiler = "simple",
    )
    targs.update(addl_targs)

    if args.debug:
        targs['log_every_n_steps'] = 1

    print0('Trainer args:', targs, file=sys.stderr)
    trainer = Trainer(**targs)

    if args.debug:
        print_dataloader(data_mod.test_dataloader())
        print_dataloader(data_mod.train_dataloader())
        print_dataloader(data_mod.val_dataloader())

    s = datetime.now()
    trainer.fit(model, data_mod)
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

def print_dataloader(dl, name=None):
    msg = list()
    if name is not None:
        msg.append(name)
    msg.append(dl.dataset.index[0])
    msg.append(dl.dataset.index[-1])
    print0(*msg)

def overall_metric(model, loader, metric):
    val = 0.0
    for idx, seqs, target, olen in loader:
        output = model(seqs)
        val += metric(target, output)
    return val

def print_args(argv=None):
    import pickle
    import ruamel.yaml as yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', type=str, help='the args.pkl file')
    args = parser.parse_args(argv)
    with open(args.pkl, 'rb') as f:
        namespace = pickle.load(f)
    yaml.main.safe_dump(vars(namespace), sys.stdout, default_flow_style=False)


def show_hparams(argv=None):
    import yaml
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='the checkpoint file to pull hparams from', type=str)
    args = parser.parse_args(argv)
    hparams = torch.load(args.checkpoint, map_location=torch.device('cpu'))['hyper_parameters']
    yaml.dump(hparams, sys.stdout)



from . import models
from .models.lit import AbstractLit

if __name__ == '__main__':
    run_lightening()
