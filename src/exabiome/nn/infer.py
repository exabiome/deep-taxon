import sys
from ..utils import check_argv, parse_logger
from .utils import process_gpus, process_model_and_dataset, process_output
import glob
import argparse
import torch



def parse_args(*addl_args, argv=None):
    """
    Parse arguments for training executable
    """
    argv = check_argv(argv)

    epi = """
    output can be used as a checkpoint
    """
    desc = "Run network training"
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('model', type=str, help='the model to run', choices=list(models._models.keys()))
    parser.add_argument('input', type=str, help='the HDF5 DeepIndex file')
    parser.add_argument('output', type=str, help='file to save model')
    parser.add_argument('-E', '--experiment', type=str, default='default', help='the experiment name')
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file', default=None)
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-L', '--load', action='store_true', default=False, help='load data into memory before running training loop')
    parser.add_argument('-U', '--umap', action='store_true', default=False, help='run 2D UMAP embedding')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('--train', action='append_const', const='train', dest='loaders', help='do inference on training data')
    parser.add_argument('--validate', action='append_const', const='validate', dest='loaders', help='do inference on validation data')
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


def process_args(args=None, return_io=False):
    """
    Process arguments for running training
    """
    if not isinstance(args, argparse.Namespace):
        args = parse_args(argv=args)

    logger = args.logger
    # set up logger
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    args.logger = logger

    targs = dict()

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

    if args.debug:
        targs['fast_dev_run'] = True

    outbase, output = process_output(args)
    if args.checkpoint is None:
        ckpt = list(glob.glob(f"{outbase}/*.ckpt"))
        if len(ckpt) > 1:
            print(f'More than one checkpoint file found in {outbase}. '
                  'Please specify checkpoint with -c', file=sys.stderr)
            sys.exit(1)
        args.checkpoint = ckpt[0]

    model, dataset, io = process_model_and_dataset(args, inference=True)
    dataset.set_classify(True)

    ret = [model, dataset, args, targs, output]
    if return_io:
        ret.append(io)

    return tuple(ret)


def run_inference():
    model, dataset, args, addl_targs, output = process_args()
    import h5py
    import numpy as np
    model.to(args.device)
    model.eval()
    n_outputs = model.classifier[-1].out_features
    f = h5py.File(output('outputs.h5'), 'w')
    emb_dset = f.create_dataset('outputs', shape=(len(dataset), n_outputs), dtype=float)
    label_dset = f.create_dataset('labels', shape=(len(dataset),), dtype=int)

    for loader_key in args.loaders:
        mask_dset = f.create_dataset(loader_key, shape=(len(dataset),), dtype=bool, fillvalue=False)
        loader = model.loaders[loader_key]
        print(f'computing outputs for {loader_key}')
        idx, outputs, labels = get_outputs(model, loader, args.device)
        order = np.argsort(idx)
        idx = idx[order]
        emb_dset[idx] = outputs[order]
        label_dset[idx] = labels[order]
        mask_dset[idx] = True

    if args.umap:
        print('Running UMAP embedding')
        from umap import UMAP
        umap = UMAP(n_components=2)
        tfm = umap.fit_transform(emb_dset[:])
        umap_dset = f.create_dataset('viz_emb', shape=(len(dataset), 2), dtype=float)
        umap_dset[:] = tfm


def get_outputs(model, loader, device):
    ret = list()
    indices = list()
    labels = list()
    it = 0
    for i, X, y, olen in loader:
        if it == 4:
            break
        ret.append(model(X.to(device)).to('cpu'))
        indices.append(i)
        labels.append(y)
        it += 1
    return torch.cat(indices).detach().numpy(), torch.cat(ret).detach().numpy(), torch.cat(labels).detach().numpy()

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
    run_inference()
