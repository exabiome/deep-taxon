import sys
import numpy as np
import os
from ..utils import check_argv, parse_logger
from .utils import process_gpus, process_model, process_output
from .train import process_config
from .loader import LazySeqDataset, get_loader, DeepIndexDataModule

from .lsf_environment import LSFEnvironment
from pytorch_lightning.plugins.environments import SLURMEnvironment


import glob
import argparse
import torch
import torch.nn as nn
import logging


def parse_args(*addl_args, argv=None):
    """
    Parse arguments for running inference
    """
    argv = check_argv(argv)

    epi = """
    output can be used as a checkpoint
    """
    desc = "Run network inference"
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('config', type=str, help='the config file used for training')
    parser.add_argument('input', type=str, help='the input file to run inference on')
    parser.add_argument('checkpoint', type=str, help='the checkpoint file to use for running inference')
    parser.add_argument('-o', '--output', type=str, help='the file to save outputs to', default=None)
    parser.add_argument('-E', '--experiment', type=str, default='default',
                        help='the experiment name to get the checkpoint from')
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-L', '--load', action='store_true', default=False,
                        help='load data into memory before running inference')
    parser.add_argument('-U', '--umap', action='store_true', default=False,
                        help='compute a 2D UMAP embedding for vizualization')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='run in debug mode i.e. only run two batches')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('-R', '--train', action='append_const', const='train', dest='loaders',
                        help='do inference on training data')
    parser.add_argument('-V', '--validate', action='append_const', const='validate', dest='loaders',
                        help='do inference on validation data')
    parser.add_argument('-S', '--test', action='append_const', const='test', dest='loaders', help='do inference on test data')
    parser.add_argument('-N', '--nonrep', action='append_const', const='nonrep', dest='loaders', help='the dataset is nonrepresentative species')
    parser.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=1)
    env_grp = parser.add_argument_group("Resource Manager").add_mutually_exclusive_group()
    env_grp.add_argument("--lsf", default='False', action='store_true', help='running in an LSF environment')
    env_grp.add_argument("--slurm", default='False', action='store_true', help='running in a SLURM environment')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    return args


def process_args(argv=None):
    """
    Process arguments for running inference
    """
    if not isinstance(argv, argparse.Namespace):
        args = parse_args(argv=argv)
    else:
        args = argv

    conf_args = process_config(args.config)
    for k, v in vars(conf_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    logger = args.logger
    # set up logger
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    args.logger = logger


    targs = dict()

    # this does not get used in this command,
    # but leave it here in case we figure out how to
    # do infernce with Lightning someday
    targs['gpus'] = process_gpus(args.gpus)
    if targs['gpus'] != 1:
        targs['distributed_backend'] = 'ddp'
        if args.slurm:
            targs['environment'] = SLURMEnvironment()
        else:
            targs['environment'] = LSFEnvironment()

    if args.debug:
        targs['fast_dev_run'] = True

    # Figure out the checkpoint file to read from
    # and where to save outputs to
    if args.output is None:
        if os.path.isdir(args.checkpoint):
            ckpt = list(glob.glob(f"{args.checkpoint}/*.ckpt"))
            if len(ckpt) == 0:
                print(f'No checkpoint file found in {args.checkpoint}', file=sys.stderr)
                sys.exit(1)
            elif len(ckpt) > 1:
                print(f'More than one checkpoint file found in {args.checkpoint}. '
                      'Please specify checkpoint with -c', file=sys.stderr)
                sys.exit(1)
            args.checkpoint = ckpt[0]
        outdir = args.checkpoint
        if outdir.endswith('.ckpt'):
            outdir = outdir[:-5]
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        args.output =  os.path.join(outdir, 'outputs.h5')

    # setting classify to so that we can get labels when
    # we load data. We do this here because we assume that
    # network is going to output features, and we want to use the
    # labels for downstream analysis
    args.classify = True

    # load the model and override batch size
    model = process_model(args, inference=True)
    model.set_inference(True)
    if args.batch_size is not None:
        model.hparams.batch_size = args.batch_size

    args.n_outputs = model.hparams.n_outputs
    args.save_seq_ids = model.hparams.window is not None

    if args.loaders is None or len(args.loaders) == 0:
        args.loaders = ['train', 'validate', 'test']

    if 'nonrep' in args.loaders:
        #dataset, io  = process_dataset(model.hparams, path=args.input)
        breakpoint()
        dataset = LazySeqDataset(path=args.input, hparams=argparse.Namespace(**model.hparams), keep_open=True)
        ldr = get_loader(dataset, batch_size=args.batch_size, distances=False)
        args.loaders = {'input': ldr}
        args.difile = dataset.difile
    else:
        data_mod = DeepIndexDataModule(args, inference=True)
        args.loaders = {'train': data_mod.train_dataloader(),
                        'validate': data_mod.val_dataloader(),
                        'test': data_mod.test_dataloader()}
        args.difile = data_mod.dataset.difile
        if args.num_workers > 0:
            data_mod.dataset.close()
        dataset = data_mod.dataset

    # return the model, any arguments, and Lighting Trainer args just in case
    # we want to use them down the line when we figure out how to use Lightning for
    # inference

    model.eval()

    ret = [model, dataset, args, targs]

    args.device = torch.device('cuda:0')

    return tuple(ret)

def run_inference(argv=None):
    """Run inference using PyTorch

    Args:
        argv: a command-line string or argparse.Namespace object to use for running inference
              If none are given, read from command-line i.e. like running argparse.ArgumentParser.parse_args
    """
    model, dataset, args, addl_targs = process_args(argv=argv)
    import h5py
    import numpy as np
    import os

    args.logger.info(f'saving outputs to {args.output}')

    n_samples = len(dataset)

    # make temporary datasets and do all I/O at the end
    tmp_emb = np.zeros(shape=(n_samples, args.n_outputs), dtype=float)
    tmp_label = np.zeros(shape=(n_samples,), dtype=int)
    tmp_olen = np.zeros(shape=(n_samples,), dtype=int)
    tmp_seq_id = None
    if args.save_seq_ids:
        tmp_seq_id = np.zeros(shape=(n_samples,), dtype=int)
    indices = list()
    masks = dict()

    model.to(args.device)
    for loader_key, loader in args.loaders.items():
        args.logger.info(f'computing outputs for {loader_key}')
        idx, outputs, labels, orig_lens, seq_ids = get_outputs(model, loader, args.device, debug=args.debug)
        order = np.argsort(idx)
        idx = idx[order]
        args.logger.info('stashing outputs, shape ' + str(outputs[order].shape))
        tmp_emb[idx] = outputs[order]
        args.logger.info('stashing labels')
        tmp_label[idx] = labels[order]
        args.logger.info('stashing orig_lens')
        tmp_olen[idx] = orig_lens
        args.logger.info('stashing mask')

        mask = np.zeros(n_samples, dtype=bool)
        mask[idx] = True
        masks[loader_key] = mask
        if args.save_seq_ids:
            args.logger.info('stashing seq_ids')
            tmp_seq_id[idx] = seq_ids[order]
        indices.append(idx)

    args.logger.info("writing data")
    f = h5py.File(args.output, 'w')
    f.create_dataset('label_names', data=dataset.label_names, dtype=h5py.special_dtype(vlen=str))

    for k,v in masks.items():
        f.create_dataset(k, data=v)

    args.logger.info("writing outputs, shape " + str(tmp_emb.shape))
    f.create_dataset('outputs', data=tmp_emb)
    args.logger.info("writing labels, shape " + str(tmp_label.shape))
    f.create_dataset('labels', data=tmp_label)
    args.logger.info("writing orig_lens, shape " + str(tmp_olen.shape))
    f.create_dataset('orig_lens', data=tmp_olen)
    if args.save_seq_ids:
        args.logger.info("writing seq_ids, shape " + str(tmp_olen.shape))
        f.create_dataset('seq_ids', data=tmp_seq_id)


    if args.umap:
        order = np.s_[:]
        if args.debug:
            indices = np.concatenate(indices)
            order = np.argsort(indices)
            indices = indices[order]
        else:
            indices = np.s_[:]
        # compute UMAP arguments for convenience
        args.logger.info('Running UMAP embedding')
        from umap import UMAP
        umap = UMAP(n_components=2)
        tfm = umap.fit_transform(tmp_emb[indices])
        umap_dset = f.create_dataset('viz_emb', shape=(n_samples, 2), dtype=float)
        umap_dset[indices] = tfm


def get_outputs(model, loader, device, debug=False, chunks=None):
    """
    Get model outputs for all samples in the given loader
    """
    outputs = list()
    indices = list()
    labels = list()
    orig_lens = list()
    seq_ids = list()
    max_batches = 100 if debug else sys.maxsize
    from tqdm import tqdm
    file = sys.stdout
    it = tqdm(loader, file=sys.stdout)
    with torch.no_grad():
        for idx, (i, X, y, olen, seq_i) in enumerate(it):
            outputs.append(model(X.to(device)).to('cpu').detach())
            indices.append(i.to('cpu').detach())
            labels.append(y.to('cpu').detach())
            orig_lens.append(olen.to('cpu').detach())
            seq_ids.append(seq_i.to('cpu').detach())
            if idx >= max_batches:
                break
    ret = (torch.cat(indices).numpy(),
           torch.cat(outputs).numpy(),
           torch.cat(labels).numpy(),
           torch.cat(orig_lens).numpy(),
           torch.cat(seq_ids).numpy())
    return ret


from . import models  # noqa: E402

if __name__ == '__main__':
    run_inference()
