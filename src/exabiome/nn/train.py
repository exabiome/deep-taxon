import sys
import os
import os.path
import warnings
import pickle
import json
import ruamel.yaml as yaml
from time import time
from tqdm import tqdm
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor, DeviceStatsMonitor

from pytorch_lightning.profiler import PyTorchProfiler

from pytorch_lightning.accelerators import GPUAccelerator, CPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin, SingleDevicePlugin, PrecisionPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
from .lsf_environment import LSFEnvironment

import torch

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    RANK = comm.Get_rank()
    SIZE = comm.Get_size()
except:
    RANK = 0
    SIZE = 1

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
        'weighted': dict(default=None, choices=[], help='weight classes in classification. options are ins, isns,ens, or phylo'),
        'ens_beta': dict(help='the value of beta to use when weighting with effective number of sample (ens)', default=0.9),
        'phylo_neighbors': dict(help='the number of neighbors to use for phylogenetic weighting', default=5),
        'n_outputs': dict(help='the number of outputs in the final layer. Ignored if --classify', default=None),
        'accumulate': dict(help='accumulate_grad_batches argument to pl.Trainer', default=1),
        'dropout_rate': dict(help='the dropout rate to use', default=0.5),
        'optimizer': dict(choices=['adam', 'lamb'], help='the optimizer to use', default='adam'),
        'lr': dict(help='the learning rate for Adam', default=0.001),
        'batch_size': dict(help='batch size', default=64),
        'lr_scheduler': dict(default='step', choices=AbstractLit.schedules, help='the learning rate schedule to use'),

        # step learning rate parameters
        'step_size': dict(help='the number of epochs between steps when using lr_scheduler="step"', default=2),
        'n_steps': dict(help='the number of steps to take when using lr_scheduler="step"', default=3),
        'step_factor': dict(help='the factor to multiple LR when using lr_scheduler="step"', default=0.1),

        # stochastic weight averaging parameters
        'swa_start': dict(help='the epoch to start stochastic weight averaging', default=7),
        'swa_anneal': dict(help='the number of epochs to anneal for when using SWA', default=1),

        'hparams': dict(help='additional hparams for the model. this should be a JSON string', default=None),
        'protein': dict(help='input contains protein sequences', default=False),
        'tnf': dict(help='input transform data to tetranucleotide frequency', default=False),
        'layers': dict(help='layers for an MLP model', default=None),
        'window': dict(type=int, help='the window size to use to chunk sequences', default=None),
        'step': dict(type=int, help='the step between windows. default is to use window size (i.e. non-overlapping chunks)', default=None),
        'n_partitions': dict(type=int, help='the number of dataset partitions.', default=1),
        'fwd_only': dict(action='store_true', help='use forward strand of sequences only', default=False),
        'classify': dict(action='store_true', help='run a classification problem', default=False),
        'manifold': dict(action='store_true', help='run a manifold learning problem', default=False),
        'bottleneck': dict(action='store_true', help='add bottleneck layer at the end of ResNet features', default=True),
        'tgt_tax_lvl': dict(choices=DeepIndexFile.taxonomic_levels, metavar='LEVEL', default='species',
                           help='the taxonomic level to predict. choices are phylum, class, order, family, genus, species'),
        'simple_clf': dict(action='store_true', help='Use a single FC layer as the classifier for ResNets', default=False),
        'dropout_clf': dict(action='store_true', help='Add dropout in FC layers as the classifier for ResNets', default=False),
    }


def print_config_options(argv=None):
    print("Available options for training config:\n")
    for k, v in get_conf_args().items():
        print(f'  {k:<15} {v["help"]} (default={v["default"]})')


def print_config_templ(argv=None):
    for k, v in get_conf_args().items():
        print(f'{k+":":<15}  # {v["help"]} (default={v["default"]})')


def process_config(conf_path, args=None):
    """
    Process arguments that are defined in the config file
    """
    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f)
    args = args or argparse.Namespace()
    for k, v in get_conf_args().items():
        conf_val = config.pop(k, v.get('default', None))
        setattr(args, k, conf_val)

    # classify by default
    if args.manifold == args.classify == False:
        args.classify = True

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
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-i', '--init', type=str, help='a checkpoint to initalize a model from', default=None)
    grp.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-T', '--test', action='store_true', help='run test data through model', default=False)
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-n', '--num_nodes', type=int, default=1, help='the number of nodes to run on')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('--fp16', action='store_true', default=False, help='use 16-bit training')
    parser.add_argument('-E', '--experiment', type=str, default='default', help='the experiment name')
    prof_grp = parser.add_mutually_exclusive_group()
    prof_grp.add_argument('--profile', action='store_true', default=False, help='profile with PyTorch Lightning profile')
    prof_grp.add_argument('--cuda_profile', action='store_true', default=False, help='profile with PyTorch CUDA profiling')
    parser.add_argument('-s', '--sanity', metavar='NBAT', nargs='?', const=True, default=False,
                        help='run NBAT batches for training and NBAT//4 batches for validation. By default, NBAT=4000')
    parser.add_argument('--lr_find', default=False, action='store_true', help='find optimal learning rate')
    grp = parser.add_argument_group('Distributed training environments').add_mutually_exclusive_group()
    grp.add_argument('--horovod', default=False, action='store_true', help='run using Horovod backend')
    grp.add_argument('--lsf', default=False, action='store_true', help='running on LSF system')
    grp.add_argument('--slurm', default=False, action='store_true', help='running on SLURM system')

    dl_grp = parser.add_argument_group('Data loading')
    dl_grp.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=0)
    dl_grp.add_argument('-y', '--pin_memory', action='store_true', default=False, help='pin memory when loading data')
    dl_grp.add_argument('-f', '--shuffle', action='store_true', default=False, help='shuffle batches when training')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='do not print arguments, model, etc.')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    process_config(args.config, args)

    return args

def process_args(args=None, return_io=False):
    """
    Process arguments for running training
    """
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args)

    args.loader_kwargs = dict()

    targs = dict(
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
    )

    targs['accumulate_grad_batches'] = args.accumulate

    targs['gpus'] = process_gpus(args.gpus)

    del args.gpus

    env = None
    if args.lsf:
        ##########################################################################################
        # Currently coding against pytorch-lightning 1.4.3
        ##########################################################################################
        if args.num_workers > 4:
            print0("num_workers (-k) > 4 can lead to hanging on Summit -- setting to 4", file=sys.stderr)
            args.num_workers = 4
        args.loader_kwargs['num_workers'] = 1           # Set as a default. This will get overridden elsewhere
        args.loader_kwargs['multiprocessing_context'] = 'spawn'
        env = LSFEnvironment()
    elif args.slurm:
        env = SLURMEnvironment()

    if env is not None:
        global RANK
        global SIZE
        try:
            RANK = env.global_rank()
            SIZE = env.world_size()
        except:
            print(">>> Could not get global rank -- setting RANK to 0 and SIZE to 1", file=sys.stderr)
            RANK = 0
            SIZE = 1

    if targs['gpus'] is not None:
        if targs['gpus'] == 1:
            targs['accelerator'] = GPUAccelerator(
                precision_plugin = NativeMixedPrecisionPlugin(16, 'cuda'),
                training_type_plugin = SingleDevicePlugin(torch.device(0))
            )
        else:
            if env is None:
                raise ValueError('Please specify environment (--lsf or --slurm) if using more than one GPU')
            parallel_devices = [torch.device(i) for i in range(torch.cuda.device_count())]
            targs['accelerator'] = GPUAccelerator(
                precision_plugin = NativeMixedPrecisionPlugin(16, 'cuda'),
                training_type_plugin = DDPPlugin(parallel_devices=parallel_devices,
                                                 cluster_environment=env,
                                                 find_unused_parameters=False)
            )
            torch.cuda.set_device(env.local_rank())
            print("---- Rank %s  -  Using GPUAccelerator with DDPPlugin" % env.global_rank(), file=sys.stderr)
            #targs['strategy'] = 'ddp_sharded'
    else:
        if env is None:
            ttp = SingleDevicePlugin(torch.device('cpu'))
        else:
            ttp = DDPPlugin(cluster_environment=env)
        targs['accelerator'] = CPUAccelerator(
            precision_plugin = PrecisionPlugin(),
            training_type_plugin = ttp
        )

    if args.sanity:
        if isinstance(args.sanity, str):
            args.sanity = int(args.sanity)
        else:
            args.sanity = 4000
        targs['limit_train_batches'] = args.sanity
        targs['limit_val_batches'] = args.sanity // 4

    if args.lr_find:
        targs['auto_lr_find'] = True
    del args.lr_find

    if args.fp16:
        targs['amp_level'] = 'O2'
        targs['precision'] = 16
    del args.fp16

    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            targs['resume_from_checkpoint'] = args.checkpoint
        else:
            warnings.warn("Ignoring -c/--checkpoint argument because {args.checkpoint} does not exist.")
            args.checkpoint = None

    if args.profile:
        #targs['profiler'] = 'advanced'
        pass
    elif args.cuda_profile:
        pass
        #targs['profiler'] = PyTorchProfiler(filename=f'pytorch_prof.{RANK:0{len(str(SIZE))}}',
        #                                    emit_nvtx=True)

    targs['replace_sampler_ddp'] = False

    args.loader_kwargs = dict()

    # make sure we are classifying if we are using adding classifier layers
    # to a resnet features model
    if args.features_checkpoint is not None:
        if args.manifold:
            raise ValueError('Cannot use manifold loss (i.e. -M) if adding classifier (i.e. -F)')
        args.classify = True

    data_mod = DeepIndexDataModule(args, keep_open=True, seed=args.seed+RANK, rank=RANK, size=SIZE)

    # if classification problem, use the number of taxa as the number of outputs
    if args.classify:
        args.n_outputs = len(data_mod.dataset.taxa_labels)

    args.input_nc = 136 if args.tnf else len(data_mod.dataset.vocab)

    model = process_model(args, taxa_table=data_mod.dataset.difile.taxa_table)

    if args.weighted is not None:
        if args.weighted == 'ens':
            weights = (1 - args.ens_beta)/(1 - args.ens_beta**data_mod.dataset.taxa_counts)
        elif args.weighted == 'isns':
            weights = np.sqrt(1/data_mod.dataset.taxa_counts)
        elif args.weighted == 'ins':
            weights = np.sqrt(1/data_mod.dataset.taxa_counts)
        elif args.weighted == 'phy':
            k = args.phylo_neighbors
            weights = np.partition(data_mod.dataset.distances, k, axis=1)[:, :k].sum(axis=1)
        else:
            raise ValueError("Unrecognized value for option 'weighted': '%s'" % args.weighted)
        model.set_class_weights(weights)

    if args.num_workers > 0:
        data_mod.dataset.close()

    ret = [model, args, targs]
    if return_io:
        ret.append(io)

    ret.append(data_mod)

    return tuple(ret)



def benchmark_pass(argv=None):
    '''Read dataset and run an epoch'''
    import numpy as np
    import traceback
    import os
    import pprint
    import torch.nn as nn
    import torch


    desc = "Read dataset and run an epoch"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config', type=str, help='the config file for this training run')
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('-N', '--num_batches', type=int, help='the number of batches to load when testing read', default=None)
    parser.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=0)
    parser.add_argument('-y', '--pin_memory', action='store_true', default=False, help='pin memory when loading data')
    parser.add_argument('-f', '--shuffle', action='store_true', default=False, help='shuffle batches when training')
    parser.add_argument('-b', '--batch_size', type=int, help='the number of workers to load data with', default=1)
    parser.add_argument('-s', '--seed', type=parse_seed, default='', help='seed for an 80/10/10 split before reading an element')
    parser.add_argument('-l', '--load', action='store_true', default=False, help='load data into memory before running training loop')

    args = parser.parse_args(argv)
    process_config(args.config, args)
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.loader_kwargs = dict()

    before = time()
    data_mod = DeepIndexDataModule(args, keep_open=True)
    after = time()

    args.input_nc = 136 if args.tnf else len(data_mod.dataset.vocab)
    if args.classify:
        args.n_outputs = len(data_mod.dataset.taxa_labels)
    model = process_model(args, taxa_table=data_mod.dataset.difile.taxa_table)

    dataset = data_mod.dataset
    print(f'Took {after - before} seconds to open {args.input}')
    difile = dataset.difile

    n_taxa = len(difile.taxa_table)
    n_seqs = len(difile.seq_table)

    n_samples = len(dataset)
    n_disc = difile.n_discarded
    wlen = args.window
    step = args.step

    if wlen is not None:
        print((f'Splitting {n_seqs} sequences (from {n_taxa} species) into {wlen} '
               f'bp windows every {step} bps produces {n_samples} samples '
               f'(after discarding {n_disc} samples).'))
    else:
        print(f'Found {n_seqs} sequences across {n_taxa} species. {n_samples} total samples')


    tr = data_mod.train_dataloader()
    # Get validation dataloader to make sure that doesn't screw up the train dataloader
    va = data_mod.val_dataloader()
    tot = len(tr)
    if args.num_batches != None:
        stop = args.num_batches - 1
        tot = args.num_batches
    else:
        stop = tot - 1

    model.to(device)
    loss = nn.CrossEntropyLoss()
    print(f'Running {tot} batches of size {args.batch_size} through model')
    before = time()
    for idx, i in tqdm(enumerate(tr), total=tot):
        result = model(i[0].to(device))
        loss(result, i[1].to(device))
        if idx == stop:
            break
    after = time()
    print(f'Took {after - before:.2f} seconds')


def print0(*msg, **kwargs):
    if RANK == 0:
        print(*msg, **kwargs)

def run_lightning(argv=None):
    '''Run training with PyTorch Lightning'''
    global RANK
    from pytorch_lightning.loggers import WandbLogger
    import numpy as np
    import traceback
    import os
    import pprint
    import wandb

    pformat = pprint.PrettyPrinter(sort_dicts=False, width=100, indent=2).pformat

    model, args, addl_targs, data_mod = process_args(parse_args(argv=argv))
    
    wandb_logger = WandbLogger(project="deep-taxon", entity='deep-taxon',
                              name=args.experiment)
    # if 'OMPI_COMM_WORLD_RANK' in os.environ or 'SLURMD_NODENAME' in os.environ:
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     RANK = comm.Get_rank()
    # else:
    #     RANK = 0
    #     print('OMPI_COMM_WORLD_RANK or SLURMD_NODENAME not set in environment -- not using MPI')

    # output is a wrapper function for os.path.join(outdir, <FILE>)
    outdir, output = process_output(args)
    check_directory(outdir)
    if not args.quiet:
        print0(' '.join(sys.argv), file=sys.stderr)
        print0("Processed Args:\n", pformat(vars(args)), file=sys.stderr)

    # save arguments
    with open(output('args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    checkpoint = None
    if args.init is not None:
        checkpoint = args.init
        link_dest = 'init.ckpt'
    elif args.checkpoint is not None:
        checkpoint = args.checkpoint
        link_dest = 'resumed_from.ckpt'

    if checkpoint is not None:
        if RANK == 0:
            print0(f'symlinking to {args.checkpoint} from {outdir}')
            dest = output(link_dest)
            src = os.path.relpath(checkpoint, start=outdir)
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

    monitor, mode = (AbstractLit.val_loss, 'min') if args.manifold else (AbstractLit.val_acc, 'max')
    callbacks = [
        ModelCheckpoint(dirpath=outdir, save_weights_only=False, save_last=True, save_top_k=3, mode=mode, monitor=monitor),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    if args.early_stop:
        callbacks.append(EarlyStopping(monitor=monitor, min_delta=0.001, patience=10, verbose=False, mode=mode))

    if args.swa:
        callbacks.append(StochasticWeightAveraging(swa_epoch_start=args.swa_start, annealing_epochs=args.swa_anneal))

    targs = dict(
        enable_checkpointing=True,
        callbacks=callbacks,
        logger = wandb_logger,#CSVLogger(save_dir=output('logs')),
        #profiler = "simple",
        num_sanity_val_steps = 0,
    )
    targs.update(addl_targs)

    if args.debug:
        targs['log_every_n_steps'] = 1
        targs['fast_dev_run'] = 10

    if not args.quiet:
        print0('Trainer args:\n', pformat(targs), file=sys.stderr)
        print0('DataLoader args\n:', pformat(data_mod._loader_kwargs), file=sys.stderr)
        print0('Model:\n', model, file=sys.stderr)

    trainer = Trainer(**targs)

    if args.debug:
        #print_dataloader(data_mod.test_dataloader())
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


def get_model_info(argv=None):
    import json
    from .loader import LazySeqDataset
    from torchinfo import summary
    import torch.nn as nn

    argv = check_argv(argv)

    epi = """
    output can be used as a checkpoint
    """
    desc = "Run network training"
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('config', type=str, help='the config file for this training run')
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    args = parser.parse_args(argv)
    process_config(args.config, args)

    dataset = LazySeqDataset(path=args.input, hparams=args, keep_open=True)
    args.input_nc = 136 if args.tnf else len(dataset.vocab)
    if args.classify:
        args.n_outputs = len(dataset.taxa_labels)

    model = process_model(args, taxa_table=dataset.difile.taxa_table)

    total_bytes = 0
    total_parameters = 0
    it = model.modules()
    next(it)
    for mod in it:
        for P in mod.parameters():
            array = P.detach().numpy()
            total_bytes += array.nbytes
            total_parameters += array.size

    print((total_bytes/1024**2), "Mb across", total_parameters, "parameters")

    print(model.fc.out_features if isinstance(model.fc, nn.Linear) else model.fc[-1].out_features)

    input_sample = torch.stack([dataset[i][1] for i in range(16)])
    summary(model, [input_sample.shape], dtypes=[torch.long])


def fix_model(argv=None):
    import json
    from .loader import LazySeqDataset
    import torch.nn as nn

    argv = check_argv(argv)

    epi = """
    DNA sequence encodings were done incorrectly before 08/25/2021. This command
    will swap in a new embedding layer that can handle new/correct encodings
    """
    desc = "Fix embedding layer for model trained with bad encoding"
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('config', type=str, help='the config file for this training run')
    parser.add_argument('init', type=str, help='the checkpoint file to fix')
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    args = parser.parse_args(argv)
    process_config(args.config, args)
    dataset = LazySeqDataset(path=args.input, hparams=args, keep_open=True)

    model = process_model(args, taxa_table=dataset.difile.taxa_table)

    ckpt = torch.load(args.init)

    output = args.init[:-4]+"fixed.ckpt"

    orig_chars = ('ACYWSKDVNTGRWSMHBN')

    new_chars =  ('ACYWSKDVNTGRMHB')

    orig_dict = {c:i for i, c in enumerate(orig_chars)}
    new_dict = {c:i for i, c in enumerate(new_chars)}

    swaps = np.array(list(zip(range(len(new_chars)), range(len(new_chars)))))

    for i in range(len(new_chars)):
        char = new_chars[i]
        old_idx = orig_dict[char]
        new_idx = new_dict[char]
        swaps[new_idx][1] = old_idx

    emb = model.embedding
    new_emb = nn.Embedding(len(new_chars), emb.embedding_dim)

    new_param = list(new_emb.parameters())[0]
    old_param = list(emb.parameters())[0]

    for i, j in swaps:
        new_param[i] = old_param[j]

    model.embedding = new_emb

    ckpt['state_dict'] = model.state_dict()

    print(f'saving checkpoint to {output}')
    torch.save(ckpt, output)


def filter_metrics(argv=None):
    import argparse
    import sys
    import pandas as pd
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=str, help='metrics.csv from Pytorch Lightning', nargs='+')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-v', '--validation', action='store_true', default=False, help='filter validation')
    grp.add_argument('-t', '--train', action='store_true', default=False, help='filter train')
    parser.add_argument('-c', '--csv', action='store_true', default=False,
                        help='write to CSV. default is to print a Pandas DataFrame')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='file to save output to. default is to print to stdout')

    args = parser.parse_args(argv)

    if not (args.validation or args.train):
        args.validation = True

    dfs = list()
    for met in args.metrics:
        df = pd.read_csv(met, na_values='', keep_default_na=False)

        if args.validation:
            columns = ['epoch', 'step', 'validation_acc', 'validation_loss']
            mask_col = 'validation_acc' if 'validation_acc' in df else 'validation_loss'
        else:
            columns = ['epoch', 'step', 'training_acc', 'training_loss']
            mask_col = 'training_acc' if 'training_acc' in df else 'training_loss'
        if mask_col not in df:
            continue

        lrdf = None
        if 'lr-AdamW' in df:
            lrdf = df.filter(['lr-AdamW'], axis=1)
            #lrdf = lrdf[np.logical_not(np.isnan(lrdf['lr-AdamW']))]
            lrdf = lrdf[np.logical_not(lrdf['lr-AdamW'].isna())]

            epdf = df.filter(['epoch'], axis=1)
            #epdf = epdf[np.logical_not(epdf['epoch'].isna()].astype(int)
            epdf = epdf[np.logical_not(np.isnan(epdf['epoch']))].astype(int)
            lrdf['epoch'] = np.unique(epdf['epoch'])

        df['epoch'] = df['epoch'].values.astype(int)

        time_df = None
        if 'time' in df:
            time_df = df[df.validation_acc.isna()]
            begin = time_df.groupby('epoch').min()[['time']]
            end = time_df.groupby('epoch').max()[['time']]
            v_df = df[df.training_acc.isna()]
            time_df = (end - begin).filter(v_df.groupby('epoch').max().index, axis=0)
            time_df['time'] /= 3600

        #mask = np.logical_not(np.isnan(df[mask_col].values))
        mask = np.logical_not(df[mask_col].isna())
        df = df.filter(columns, axis=1)[mask]

        df = df.drop('step', axis=1)
        if lrdf is not None:
            df = df.set_index('epoch').merge(lrdf.set_index('epoch'), left_index=True, right_index=True)
        if time_df is not None:
            df = df.merge(time_df, left_index=True, right_index=True)
        dfs.append(df)

    df = pd.concat(dfs)

    out = sys.stdout
    if args.output is not None:
        out = open(args.output, 'w')

    if args.csv:
        df.to_csv(out)
    else:
        pd.set_option('display.max_rows', None)
        print(df, file=out)


from . import models
from .models.lit import AbstractLit

if __name__ == '__main__':
    run_lightening()
