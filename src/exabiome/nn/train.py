import sys
import os
import os.path
import pickle
import json
import ruamel.yaml as yaml
from datetime import datetime
import numpy as np
from ..utils import parse_seed, check_argv, parse_logger, check_directory
from .utils import process_gpus, process_model, process_output
from .loader import add_dataset_arguments, DeepIndexDataModule
from ..sequence import DeepIndexFile
from hdmf.utils import docval

import argparse
import logging


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

from pytorch_lightning.accelerators import GPUAccelerator, CPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin, SingleDevicePlugin, PrecisionPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
from .lsf_environment import LSFEnvironment

import torch

def parse_train_size(string):
    ret = float(string)
    if ret > 1.0:
        ret = int(ret)
    return ret


def get_conf_args():
    return {
        'model': dict(help='the model to run. see show-models for a list of available models', choices=list(models._models.keys()), default='resnet18'),
        'seed': dict(type=parse_seed, default='', help='seed to use for train-test split'),
        'downsample': dict(type=float, default=None, help='amount to downsample dataset to'),
        'weighted': dict(default=None, choices=[], help='weight classes in classification. options are ins, isns, or ens'),
        'ens_beta': dict(help='the value of beta to use when weighting with effective number of sample (ens)', default=0.9),
        'n_outputs': dict(help='the number of outputs in the final layer. Ignored if --classify', default=32),
        'accumulate': dict(help='accumulate_grad_batches argument to pl.Trainer', default=1),
        'dropout_rate': dict(help='the dropout rate to use', default=0.5),
        'optimizer': dict(choices=['adam', 'lamb'], help='the optimizer to use', default='adam'),
        'lr': dict(help='the learning rate for Adam', default=0.001),
        'lr_scheduler': dict(default='adam', choices=AbstractLit.schedules, help='the learning rate schedule to use'),
        'batch_size': dict(help='batch size', default=64),
        'hparams': dict(help='additional hparams for the model. this should be a JSON string', default=None),
        'protein': dict(help='input contains protein sequences', default=False),
        'window': dict(type=int, help='the window size to use to chunk sequences', default=None),
        'step': dict(type=int, help='the step between windows. default is to use window size (i.e. non-overlapping chunks)', default=None),
        'fwd_only': dict(action='store_true', help='use forward strand of sequences only', default=False),
        'classify': dict(action='store_true', help='run a classification problem', default=False),
        'manifold': dict(action='store_true', help='run a manifold learning problem', default=False),
        'tgt_tax_lvl': dict(choices=DeepIndexFile.taxonomic_levels, metavar='LEVEL', default='species',
                           help='the taxonomic level to predict. choices are phylum, class, order, family, genus, species'),
    }


def print_config_options(argv=None):
    print("Available options for training config:\n")
    for k, v in get_conf_args().items():
        print(f'  {k:<15} {v["help"]} (default={v["default"]})')


def print_config_templ(argv=None):
    for k, v in get_conf_args().items():
        print(f'{k+":":<15}  # {v["help"]} (default={v["default"]})')


def process_config(conf_path, args=None):
    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f)
    args = args or argparse.Namespace()
    for k, v in get_conf_args().items():
        conf_val = config.pop(k, v.get('default', None))
        setattr(args, k, conf_val)
    return args


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
    parser.add_argument('config', type=str, help='the config file for this training run')
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('output', type=str, help='file to save model', default=None)

    ##################################################
    # Add dataset-specific arguments, like tgt_tax_lvl
    ##################################################

    parser.add_argument('-F', '--features_checkpoint', type=str, help='a checkpoint file for previously trained features', default=None)
    parser.add_argument('-l', '--load', action='store_true', default=False, help='load data into memory before running training loop')
    parser.add_argument('--early_stop', action='store_true', default=False, help='stop early if validation loss does not improve')
    parser.add_argument('--swa', action='store_true', default=False, help='use stochastic weight averaging')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-T', '--test', action='store_true', help='run test data through model', default=False)
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-n', '--num_nodes', type=int, default=1, help='the number of nodes to run on')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('--fp16', action='store_true', default=False, help='use 16-bit training')
    parser.add_argument('-E', '--experiment', type=str, default='default', help='the experiment name')
    parser.add_argument('--profile', action='store_true', default=False, help='profile with PyTorch Lightning profile')
    parser.add_argument('--sanity', action='store_true', default=False, help='copy response data into input data')
    parser.add_argument('--lr_find', default=False, action='store_true', help='find optimal learning rate')
    grp = parser.add_argument_group('Distributed training environments').add_mutually_exclusive_group()
    grp.add_argument('--horovod', default=False, action='store_true', help='run using Horovod backend')
    grp.add_argument('--lsf', default=False, action='store_true', help='running on LSF system')
    grp.add_argument('--slurm', default=False, action='store_true', help='running on SLURM system')

    dl_grp = parser.add_argument_group('Data loading')
    dl_grp.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=1)
    dl_grp.add_argument('-y', '--pin_memory', action='store_true', default=False, help='pin memory when loading data')
    dl_grp.add_argument('-f', '--shuffle', action='store_true', default=False, help='shuffle batches when training')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    process_config(args.config, args)

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

    input_nc = 18
    if args.protein:
        input_nc = 26
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

    data_mod = DeepIndexDataModule(args, keep_open=True)

    # if classification problem, use the number of taxa as the number of outputs
    if args.classify:
        args.n_outputs = len(data_mod.dataset.taxa_labels)
    model = process_model(args, taxa_table=data_mod.dataset.difile.taxa_table)

    if args.weighted is not None:
        if args.weighted == 'ens':
            weights = (1 - args.ens_beta)/(1 - args.ens_beta**data_mod.dataset.taxa_counts)
        elif args.weighted == 'isns':
            weights = np.sqrt(1/data_mod.dataset.taxa_counts)
        elif args.weighted == 'ins':
            weights = np.sqrt(1/data_mod.dataset.taxa_counts)
        else:
            raise ValueError("Unrecognized value for option 'weighted': '%s'" % args.weighted)
        model.set_class_weights(weights)

    data_mod.dataset.close()

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
                                                 cluster_environment=env, num_nodes=args.num_nodes)
            )
            torch.cuda.set_device(env.local_rank())
            print("---- Rank %s  -  Using GPUAccelerator with DDPPlugin" % env.global_rank(), file=sys.stderr)
    else:
        targs['accelerator'] = CPUAccelerator(
            precision_plugin = PrecisionPlugin(),
            training_type_plugin = DDPPlugin(cluster_environment=env, num_nodes=args.num_nodes)
        )

    if args.sanity:
        targs['limit_train_batches'] = 40
        targs['limit_val_batches'] = 5
        targs['max_epochs'] = min(args.epochs, 5)

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
    global RANK

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
    RANK = addl_targs['accelerator'].training_type_plugin.cluster_environment.global_rank()

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

    callbacks = [ModelCheckpoint(dirpath=outdir, save_weights_only=False, save_last=True, save_top_k=3, monitor=AbstractLit.val_loss)]

    if args.early_stop:
        callbacks.append(EarlyStopping(monitor=AbstractLit.val_loss, min_delta=0.00, patience=3, verbose=False, mode='min'))

    if args.swa:
        callbacks.append(StochasticWeightAveraging(swa_epoch_start=5, annealing_epochs=5))

    targs = dict(
        checkpoint_callback=True,
        callbacks=callbacks,
        logger = CSVLogger(save_dir=output('logs')),
        profiler = "simple",
    )
    targs.update(addl_targs)

    if args.debug:
        targs['log_every_n_steps'] = 1

    print0('Trainer args:', targs, file=sys.stderr)
    print0('Model:\n', model, file=sys.stderr)

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
