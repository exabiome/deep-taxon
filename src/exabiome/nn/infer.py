import sys
import os
from ..utils import check_argv, parse_logger
from .utils import process_gpus, process_model, process_output
from .loader import process_dataset, get_loader
import glob
import argparse
import torch
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
    parser.add_argument('model', type=str, choices=list(models._models.keys()),
                        metavar='MODEL',
                        help='the model type to run inference with')
    parser.add_argument('checkpoint', type=str, help='read the checkpoint from the given checkpoint file')
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
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=None)
    parser.add_argument('-I', '--input', type=str, help='the HDF5 DeepIndex file used to train the model', default=None)
    parser.add_argument('-R', '--train', action='append_const', const='train', dest='loaders',
                        help='do inference on training data')
    parser.add_argument('-V', '--validate', action='append_const', const='validate', dest='loaders',
                        help='do inference on validation data')
    parser.add_argument('-S', '--test', action='append_const', const='test', dest='loaders', help='do inference on test data')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    if not (args.input is None or args.loaders is None):   # don't pass an input file and use the TVT data
        print('-I/--input cannot be used with any of -R/--train, -V/--validate, -S/--test', file=sys.stderr)
        sys.exit(1)

    return args


def process_args(argv=None):
    """
    Process arguments for running inference
    """
    if not isinstance(argv, argparse.Namespace):
        args = parse_args(argv=argv)
    else:
        args = argv

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
        targs['distributed_backend'] = 'dp'
    if args.gpus:
        dev = "cuda:0"
    else:
        dev = "cpu"
    print(f'using {dev} device')
    args.device = torch.device(dev)
    del args.gpus
    args.input_nc = 5

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

    if args.input is not None:                          # if an input file is passed in, use that first
        dset, io  = process_dataset(model.hparams, path=args.input, inference=True)
        ldr = get_loader(dset, distances=False)
        args.loaders = {'input': ldr}
        args.difile = dset.difile
    elif args.loader is None:                           # if an input file is not passed in, do all TVT data
        args.loaders = {'train': model.train_dataloader(),
                        'validate': model.validate_dataloader(),
                        'test': model.test_dataloader()}
        args.difile = model.dataset.difile

    # return the model, any arguments, and Lighting Trainer args just in case
    # we want to use them down the line when we figure out how to use Lightning for
    # inference
    ret = [model, args, targs]

    return tuple(ret)

from .. import command

@command('infer')
def run_inference(argv=None):
    """Run inference using PyTorch

    Args:
        argv: a command-line string or argparse.Namespace object to use for running inference
              If none are given, read from command-line i.e. like running argparse.ArgumentParser.parse_args
    """
    model, args, addl_targs = process_args(argv=argv)
    import h5py
    import numpy as np
    model.to(args.device)
    model.eval()
    n_outputs = model.hparams.n_outputs
    n_samples = len(args.difile)

    args.logger.info(f'saving outputs to {args.output}')
    f = h5py.File(args.output, 'w')

    emb_dset = f.create_dataset('outputs', shape=(n_samples, n_outputs), dtype=float)
    label_dset = f.create_dataset('labels', shape=(n_samples,), dtype=int)
    olen_dset = f.create_dataset('orig_lens', shape=(n_samples,), dtype=int)
    f.create_dataset('taxon_id', data=args.difile.taxa_table['rep_taxon_id'][:])
    seq_id_dset = None
    if model.hparams.window is not None:
        seq_id_dset = f.create_dataset('seq_ids', shape=(n_samples,), dtype=int)

    for loader_key, loader in args.loaders.items():
        mask_dset = f.create_dataset(loader_key, shape=(n_samples,), dtype=bool, fillvalue=False)
        args.logger.info(f'computing outputs for {loader_key}')
        idx, outputs, labels, orig_lens, seq_ids = get_outputs(model, loader, args.device, debug=args.debug)
        order = np.argsort(idx)
        idx = idx[order]
        args.logger.info('writing outputs')
        emb_dset[idx] = outputs[order]
        args.logger.info('writing labels')
        label_dset[idx] = labels[order]
        args.logger.info('writing orig_lens')
        olen_dset[idx] = orig_lens
        args.logger.info('writing mask')
        mask_dset[idx] = True
        if seq_id_dset is not None:
            seq_id_dset[idx] = seq_ids[order]

    if args.umap:
        # compute UMAP arguments for convenience
        args.logger.info('Running UMAP embedding')
        from umap import UMAP
        umap = UMAP(n_components=2)
        tfm = umap.fit_transform(emb_dset[:])
        umap_dset = f.create_dataset('viz_emb', shape=(n_samples, 2), dtype=float)
        umap_dset[:] = tfm


def get_outputs(model, loader, device, debug=False):
    """
    Get model outputs for all samples in the given loader
    """
    ret = list()
    indices = list()
    labels = list()
    orig_lens = list()
    seq_ids = list()
    idx = 1
    from tqdm import tqdm
    if debug:
        it = tqdm([next(loader[0])])
    else:
        it = tqdm(loader)
    for i, X, y, olen, seq_i in it:
        idx += 1
        ret.append(model(X.to(device)).to('cpu').detach())
        indices.append(i.to('cpu').detach())
        labels.append(y.to('cpu').detach())
        orig_lens.append(olen.to('cpu').detach())
        seq_ids.append(seq_i.to('cpu').detach())
    ret = (torch.cat(indices).numpy(),
           torch.cat(ret).numpy(),
           torch.cat(labels).numpy(),
           torch.cat(orig_lens).numpy(),
           torch.cat(seq_ids).numpy())
    return ret


from . import models  # noqa: E402

if __name__ == '__main__':
    run_inference()
