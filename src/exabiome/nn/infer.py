import sys
from ..utils import check_argv, parse_logger
from .utils import process_gpus, process_model, process_output
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
    desc = "Run network training"
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('model', type=str, choices=list(models._models.keys()),
                        help='the model type to run inference with')
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file used to train the model')
    parser.add_argument('output', type=str, help='directory to save model outputs to')
    parser.add_argument('-E', '--experiment', type=str, default='default',
                        help='the experiment name to get the checkpoint from')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='read the checkpoint from the given checkpoint file')
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-L', '--load', action='store_true', default=False,
                        help='load data into memory before running inference')
    parser.add_argument('-U', '--umap', action='store_true', default=False,
                        help='compute a 2D UMAP embedding for vizualization')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='run in debug mode i.e. only run two batches')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=None)
    parser.add_argument('--train', action='append_const', const='train', dest='loaders',
                        help='do inference on training data')
    parser.add_argument('--validate', action='append_const', const='validate', dest='loaders',
                        help='do inference on validation data')
    parser.add_argument('--test', action='append_const', const='test', dest='loaders', help='do inference on test data')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    if args.loaders is None:
        args.loaders = ['train', 'validate', 'test']

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
    outbase, output = process_output(args)
    if args.checkpoint is None:
        ckpt = list(glob.glob(f"{outbase}/*.ckpt"))
        if len(ckpt) > 1:
            print(f'More than one checkpoint file found in {outbase}. '
                  'Please specify checkpoint with -c', file=sys.stderr)
            sys.exit(1)
        args.checkpoint = ckpt[0]
    args.output = '%s.outputs.h5' % args.checkpoint[:-5]

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

    # return the model, any arguments, and Lighting Trainer args just in case
    # we want to use them down the line when we figure out how to use Lightning for
    # inference
    ret = [model, args, targs]

    return tuple(ret)


def run_inference(argv=None):
    """
    Run inference

    Args:
        argv: a command-line string or argparse.Namespace object to use for running inference
              If none are given, read from command-line i.e. like running argparse.ArgumentParser.parse_args
    """
    model, args, addl_targs = process_args(argv=argv)
    import h5py
    import numpy as np
    model.to(args.device)
    model.eval()
    n_outputs = model.classifier[-1].out_features
    n_samples = len(model.train_dataloader().dataset) +\
        len(model.val_dataloader().dataset) +\
        len(model.test_dataloader().dataset)

    f = h5py.File(args.output, 'w')

    emb_dset = f.create_dataset('outputs', shape=(n_samples, n_outputs), dtype=float)
    label_dset = f.create_dataset('labels', shape=(n_samples,), dtype=int)
    olen_dset = f.create_dataset('orig_lens', shape=(n_samples,), dtype=int)
    f.create_dataset('taxon_id', data=model.train_dataloader().dataset.difile.taxa_table['taxon_id'][:])

    for loader_key in args.loaders:
        mask_dset = f.create_dataset(loader_key, shape=(n_samples,), dtype=bool, fillvalue=False)
        loader = model.loaders[loader_key]
        args.logger.info(f'computing outputs for {loader_key}')
        idx, outputs, labels, orig_lens = get_outputs(model, loader, args.device, debug=args.debug)
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
    idx = 1
    from tqdm import tqdm
    for i, X, y, olen in tqdm(loader):
        idx += 1
        ret.append(model(X.to(device)).to('cpu').detach())
        indices.append(i.to('cpu').detach())
        labels.append(y.to('cpu').detach())
        orig_lens.append(olen.to('cpu').detach())
        if debug:
            break
    ret = (torch.cat(indices).numpy(),
           torch.cat(ret).numpy(),
           torch.cat(labels).numpy(),
           torch.cat(orig_lens).numpy())
    return ret


from . import models  # noqa: E402

if __name__ == '__main__':
    run_inference()
