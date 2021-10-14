import sys
import numpy as np
import os
from ..utils import check_argv, parse_logger, ccm
from .utils import process_gpus, process_model, process_output
from .train import process_config
from .loader import LazySeqDataset, get_loader, DeepIndexDataModule

from .lsf_environment import LSFEnvironment
from pytorch_lightning.plugins.environments import SLURMEnvironment


import glob
import h5py
import argparse
import torch
from torch.utils.data import Subset
from torch.cuda.amp import autocast
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
    parser.add_argument('-f', '--features', action='store_true', help='drop classifier from ResNet model before inference', default=False)
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
    parser.add_argument('-M', '--in_memory', default=False, action='store_true', help='collect all batches in memory before writing to disk')
    parser.add_argument('-B', '--n_batches', type=int, default=100, help='the number of batches to accumulate between each write to disk')
    parser.add_argument('-s', '--start', type=int, help='sample index to start at', default=0)
    env_grp = parser.add_argument_group("Resource Manager").add_mutually_exclusive_group()
    env_grp.add_argument("--lsf", default=False, action='store_true', help='running in an LSF environment')
    env_grp.add_argument("--slurm", default=False, action='store_true', help='running in a SLURM environment')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    args.init = args.checkpoint

    return args


def split(dset_len, size, rank):
    q, r = divmod(dset_len, size)
    if rank < r:
        q += 1
        b = rank*q
        return np.arange(b, b+q)
    else:
        offset = (q+1)*r
        b = (rank - r)*q + offset
        return np.arange(b, b+q)


def process_args(args, size=1, rank=0, comm=None):
    """
    Process arguments for running inference
    """
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
    elif args.lsf:
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

    # remove ResNet features
    if args.features:
        if 'ResNet' not in model.__class__.__name__:
            raise ValueError("Cannot use -f without ResNet model - got %s" % model.__class__.__name__)
        from .models.resnet import ResNetFeatures
        args.n_outputs = model.fc.in_features
        model = ResNetFeatures(model)

    if args.loaders is None or len(args.loaders) == 0:
        args.loaders = ['train', 'validate', 'test']

    if 'nonrep' in args.loaders:
        if size > 1:
            dataset = LazySeqDataset(path=args.input, hparams=argparse.Namespace(**model.hparams), keep_open=True, comm=comm)
        else:
            dataset = LazySeqDataset(path=args.input, hparams=argparse.Namespace(**model.hparams), keep_open=True)
        tmp_dset = dataset
        if args.start > 0:
            tmp_dset = Subset(tmp_dset, np.arange(args.start, len(dataset)))

        if size > 1:
            subset = split(len(tmp_dset), size, rank)
            tmp_dset = Subset(tmp_dset, subset)
            args.logger.info(f'rank {rank} - processing {len(subset)} samples subset')

        ldr = get_loader(tmp_dset, batch_size=args.batch_size, distances=False)
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

    if size > 1:
        if 'environment' not in targs:
            print(f'rank {rank} - Please specify --lsf or --slurm if running distributed', file=sys.stderr)
            sys.exit(1)
        args.device = torch.device('cuda:%d' % targs['environment'].local_rank())
    else:
        args.device = torch.device('cuda:0')

    return tuple(ret)


def run_inference(argv=None):
    """Run inference using PyTorch

    Args:
        argv: a command-line string or argparse.Namespace object to use for running inference
              If none are given, read from command-line i.e. like running argparse.ArgumentParser.parse_args
    """

    args = parse_args(argv=argv)
    RANK = 0
    SIZE = 1
    COMM = None
    f_kwargs = dict()
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        # load this after so we can get usage
        # statement without having to load MPI
        from mpi4py import MPI

        COMM = MPI.COMM_WORLD
        RANK = COMM.Get_rank()
        SIZE = COMM.Get_size()
        if SIZE > 1:
            f_kwargs['driver'] = 'mpio'
            f_kwargs['comm'] = COMM
    else:
        args.logger.info('OMPI_COMM_WORLD_RANK not set in environment -- not using MPI')



    model, dataset, args, addl_targs = process_args(args, size=SIZE, rank=RANK)

    if args.in_memory:
        if SIZE > 1:
            print("Cannot do in-memory inference with MPI", file=sys.stderr)
            exit(1)
        args.logger.info(f'running in-memory inference')
        in_memory_inference(model, dataset, args, addl_targs)
    else:
        if SIZE > 1:
            args.logger.info(f'running parallel chunked inference')
        else:
            args.logger.info(f'running serial chunked inference')
        parallel_chunked_inference(model, dataset, args, addl_targs, COMM, SIZE, RANK, f_kwargs)


def parallel_chunked_inference(model, dataset, args, addl_targs, comm, size, rank, fkwargs):
    n_samples = len(dataset)
    f = h5py.File(args.output, 'w', **fkwargs)
    outputs = f.require_dataset('outputs', shape=(n_samples, args.n_outputs), dtype=float)
    labels = f.require_dataset('labels', shape=(n_samples,), dtype=int)
    orig_lens = f.require_dataset('orig_lens', shape=(n_samples,), dtype=int)
    seq_ids = f.require_dataset('seq_ids', shape=(n_samples,), dtype=int)

    model.to(args.device)

    if size > 1: comm.Barrier()

    for loader_key, loader in args.loaders.items():
        if rank == 0: args.logger.info(f'computing outputs for {loader_key}')
        mask = f.require_dataset(loader_key, shape=(n_samples,), dtype=bool, fillvalue=False)
        for idx, _outputs, _labels, _orig_lens, _seq_ids in get_outputs(model, loader, args.device, debug=args.debug, chunks=args.n_batches, prog_bar=rank==0):
            order = np.argsort(idx)
            idx = idx[order]
            with ccm(size > 1, outputs.collective):
                outputs[idx] = _outputs[order]
            with ccm(size > 1, labels.collective):
                labels[idx] = _labels[order]
            with ccm(size > 1, orig_lens.collective):
                orig_lens[idx] = _orig_lens[order]
            with ccm(size > 1, seq_ids.collective):
                seq_ids[idx] = _seq_ids[order]
            with ccm(size > 1, mask.collective):
                mask[idx] = True

    if size > 1: comm.Barrier()

    f.close()
    if rank == 0: args.logger.info(f'closing {args.output}')


def get_outputs(model, loader, device, debug=False, chunks=None, prog_bar=True):
    """
    Get model outputs for all samples in the given loader

    Parameters
    ----------
    model: torch.Module
        the PyTorch model to get outputs for

    loader: torch.DataLoader
        the PyTorch DataLoader to get data from

    device: torch.device
        the PyTorch device to run computation on

    debug: bool
        run 100 batches and return

    chunks: int
        if not None, return a subsets of batches of size *chunks* as a
        generator

    """
    indices, outputs, labels, orig_lens, seq_ids = [], [], [], [], []
    max_batches = 100 if debug else sys.maxsize
    from tqdm import tqdm
    file = sys.stdout
    it = loader
    if prog_bar:
        it = tqdm(it, file=sys.stdout)
    with torch.no_grad():
        for idx, (i, X, y, olen, seq_i) in enumerate(it):
            X = X.to(device)
            with autocast():
                out = model(X).to('cpu').detach()
            outputs.append(out)
            indices.append(i.to('cpu').detach())
            labels.append(y.to('cpu').detach())
            orig_lens.append(olen.to('cpu').detach())
            seq_ids.append(seq_i.to('cpu').detach())
            if idx >= max_batches:
                break
            if chunks and (idx+1) % chunks == 0:
                yield cat(indices, outputs, labels, orig_lens, seq_ids)
                indices, outputs, labels, orig_lens, seq_ids = [], [], [], [], []
    if chunks is None:
        return cat(indices, outputs, labels, orig_lens, seq_ids)
    else:
        if len(indices) > 0:
            yield cat(indices, outputs, labels, orig_lens, seq_ids)


def cat(indices, outputs, labels, orig_lens, seq_ids):
    ret = (torch.cat(indices).numpy(),
           torch.cat(outputs).numpy(),
           torch.cat(labels).numpy(),
           torch.cat(orig_lens).numpy(),
           torch.cat(seq_ids).numpy())
    return ret


def in_memory_inference(model, dataset, args, addl_targs):

    from pkg_resources import resource_filename
    from hdmf.common import load_namespaces, get_class, VectorData, EnumData
    from hdmf.backends.hdf5 import HDF5IO

    load_namespaces(os.path.join(resource_filename(__name__, '../hdmf-ml'), 'schema', 'ml', 'namespace.yaml'))
    ResultsTable = get_class('ResultsTable', 'hdmf-ml')
    ClassProbability = get_class('ClassProbability', 'hdmf-ml')
    ClassLabel = get_class('ClassLabel', 'hdmf-ml')
    TrainValidationTestSplit = get_class('TrainValidationTestSplit', 'hdmf-ml')
    EmbeddedValues = get_class('EmbeddedValues', 'hdmf-ml')

    n_samples = len(dataset)
    # make temporary datasets and do all I/O at the end
    output_data = np.zeros(shape=(n_samples, args.n_outputs), dtype=float)
    predicted_class_data = np.zeros(shape=(n_samples,), dtype=int)
    true_label_data = np.zeros(shape=(n_samples,), dtype=int)
    orig_lens_data = np.zeros(shape=(n_samples,), dtype=int)
    tvt_split_data = np.zeros(shape=(n_samples,), dtype=int)
    tvt_split_elements_data = ['train', 'validate', 'test']  # TODO these will eventually be set in the TVT data_type
    true_label_elements_data = dataset.label_names  # TODO inspect this

    tmp_seq_id = None
    if args.save_seq_ids:
        tmp_seq_id = np.zeros(shape=(n_samples,), dtype=int)
    indices = list()

    model.to(args.device)
    for loader_key, loader in args.loaders.items():
        # separate loaders for train, validate, test
        args.logger.info(f'computing outputs for {loader_key}')
        idx, outputs, labels, orig_lens, seq_ids = get_outputs(model, loader, args.device, debug=args.debug)
        order = np.argsort(idx)
        idx = idx[order]
        args.logger.info('stashing outputs, shape ' + str(outputs[order].shape))
        output_data[idx] = outputs[order]
        if args.classify:
            pred_class = np.amax(outputs[order], axis=1)
            args.logger.info('stashing predicted class, shape ' + str(pred_class.shape))
            predicted_class_data[idx] = pred_class
        args.logger.info('stashing labels')
        true_label_data[idx] = labels[order]
        args.logger.info('stashing orig_lens')
        orig_lens_data[idx] = orig_lens
        args.logger.info('stashing mask')
        assert loader_key in tvt_split_elements_data
        tvt_split_data[idx] = tvt_split_elements_data.index(loader_key)
        if args.save_seq_ids:
            args.logger.info('stashing seq_ids')
            tmp_seq_id[idx] = seq_ids[order]
        indices.append(idx)

    args.logger.info("writing data")

    # if args.umap:
    #     order = np.s_[:]
    #     if args.debug:
    #         indices = np.concatenate(indices)
    #         order = np.argsort(indices)
    #         indices = indices[order]
    #     else:
    #         indices = np.s_[:]
    #     # compute UMAP arguments for convenience
    #     args.logger.info('Running UMAP embedding')
    #     from umap import UMAP
    #     umap = UMAP(n_components=2)
    #     tfm = umap.fit_transform(tmp_output[indices])
    #     umap_dset = np.zeros(shape=(n_samples, 2), dtype=float)
    #     umap_dset[indices] = tfm

    # create and set up results table
    columns = list()

    args.logger.info("writing tvt_split, shape " + str(tvt_split_data.shape))
    tvt_split_elements_col = VectorData(
        name='tvt_split_elements',
        description=('Fixed set of elements referenced by tvt_split EnumData column. Usually has values '
                     '"train", "validate", "test"'),
        data=tvt_split_elements_data,
    )
    columns.append(tvt_split_elements_col)

    tvt_split_col = TrainValidationTestSplit(
        name='tvt_split',
        description='A column to indicate if a sample was used for training, testing or validation',
        data=tvt_split_data,
        elements=tvt_split_elements_col,
    )
    columns.append(tvt_split_col)

    args.logger.info("writing orig_length, shape " + str(orig_lens_data.shape))
    orig_lens_col = VectorData(
        name='orig_length',
        description='Original lengths',
        data=orig_lens_data,
    )
    columns.append(orig_lens_col)

    args.logger.info("writing true_label, shape " + str(true_label_data.shape))
    true_label_elements_col = VectorData(
        name='true_label_elements',
        description='Fixed set of elements referenced by true_label EnumData column',
        data=true_label_elements_data,
    )
    columns.append(true_label_elements_col)

    true_label_col = EnumData(
        name='true_label',
        description='A column to store the true label for each sample',
        data=true_label_data,
        elements=true_label_elements_col,
        # TODO link to true label elements of original taxa table??
        # elements=dataset.difile.taxa_table[dataset.difile.label_key].elements
    )
    columns.append(true_label_col)

    if args.classify:
        args.logger.info("writing predicted_proba, shape " + str(output_data.shape))
        predicted_proba_col = ClassProbability(
            name='predicted_proba',
            description='A column to store the class probability for each class across the samples',
            data=output_data,
            training_labels=true_label_col,
        )
        columns.append(predicted_proba_col)

        args.logger.info("writing predicted_class, shape " + str(predicted_class_data.shape))
        predicted_class_col = ClassLabel(
            name='predicted_class',
            description='A column to store which class a sample was classified as',
            data=predicted_class_data,
            training_labels=true_label_col,
        )
        columns.append(predicted_class_col)
    else:
        args.logger.info("writing embedding, shape " + str(output_data.shape))
        embedding_col = EmbeddedValues(
            name='embedding',
            description='A column to store embeddings for each sample',
            data=output_data,
        )
        columns.append(embedding_col)

    if args.save_seq_ids:
        args.logger.info("writing seq_ids, shape " + str(tmp_seq_id.shape))
        embedding_col = VectorData(
            name='seq_id',
            description='Sequential IDs',
            data=tmp_seq_id
        )
        columns.append(embedding_col)

    # if args.umap:
    #     args.logger.info("writing viz_emb, shape " + str(umap_dset.shape))
    #     viz_emb_col = VectorData(
    #         name='viz_emb',
    #         description='UMAP visualization embedding',
    #         data=umap_dset
    #     )
    #     columns.append(viz_emb_col)

    table = ResultsTable(name='results', description='ML results', columns=columns)

    args.logger.info(f'saving outputs to {args.output}')
    with HDF5IO(args.output, 'w') as io:
        io.write(table)


from . import models  # noqa: E402

if __name__ == '__main__':
    run_inference()
