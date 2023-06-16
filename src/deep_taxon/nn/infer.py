import sys
import warnings
import numpy as np
import os
from time import time
from functools import partial
from ..utils import check_argv, parse_logger, ccm, distsplit
from .utils import process_gpus, process_model, process_output
from .train import process_config
from .loader import LazySeqDataset, get_loader, DeepIndexDataModule

from hdmf.common import get_hdf5io

from .lsf_environment import LSFEnvironment
from pytorch_lightning.plugins.environments import SLURMEnvironment


import glob
import h5py
import argparse
import torch
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
    parser.add_argument('--force', action='store_true', help='overwrite existing outputs file', default=False)
    parser.add_argument('-f', '--resnet_features', action='store_true', help='drop classifier from ResNet model before inference', default=False)
    parser.add_argument('-F', '--features', action='store_true', help='outputs are features i.e. do not softmax and compute predictions', default=False)
    parser.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='run in debug mode i.e. only run two batches')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)

    parser.add_argument('-N', '--nonrep', action='append_const', const='nonrep', dest='loaders', help='the dataset is nonrepresentative species')
    parser.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=1)
    parser.add_argument('-M', '--in_memory', default=False, action='store_true', help='collect all batches in memory before writing to disk')
    parser.add_argument('-B', '--n_batches', type=int, default=100, help='the number of batches to accumulate between each write to disk or aggregation')
    # parser.add_argument('-a', '--aggregate', action='store_true', help='aggregate chunks within sequences', default=False)
    parser.add_argument('-S', '--n_seqs', type=int, default=500, help='the number of sequences to aggregate chunks for between each write to disk')
    parser.add_argument('-p', '--maxprob', metavar='TOPN', default=2, type=int,
                        help='store the top TOPN probablities of each output. By default, TOPN=1')
    parser.add_argument("-H", "--hierarchy", default=False, action='store_true', help='force predictions to follow probability hierarchy')
    parser.add_argument('-c', '--save_chunks', action='store_true', help='do store network outputs for each chunk', default=False)
    parser.add_argument('-O', '--overlap', action='store_true',  default=False,
                        help='overlap with step size used during training. default is to use nonoverlapping windows')

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


def process_args(args, comm=None):
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
        logger = logging.getLogger('inference')
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    args.logger = logger


    rank = 0
    size = 1
    local_rank = 0
    env = None

    if args.slurm:
        env = SLURMEnvironment()
    elif args.lsf:
        env = LSFEnvironment()

    if env is not None:
        local_rank = env.local_rank()
        rank = env.global_rank()
        size = env.world_size()

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

    if os.path.exists(args.output) and not args.force:
        print(f'Output file {args.output} exists. Exiting. Use --force to override this behavior', file=sys.stderr)
        sys.exit(1)

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
    if args.resnet_features:
        if 'ResNet' not in model.__class__.__name__:
            raise ValueError("Cannot use -f without ResNet model - got %s" % model.__class__.__name__)
        from .models.resnet import ResNetFeatures
        args.n_outputs = model.fc.in_features if isinstance(model.fc, nn.Linear) else model.fc[0].in_features
        model = ResNetFeatures(model)
        args.features = True

    io = get_hdf5io(args.input, 'r')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        args.difile = io.read()

    args.difile.set_label_key(args.tgt_tax_lvl)

    # read the taxonomy table so we can compute higher-order taxonomic probabilities
    tt = args.difile.taxa_table
    _load = lambda x: x[:]
    for col in tt.columns:
        col.transform(_load)
    tt = tt.to_dataframe(index=True)

    args.total_seqs = len(args.difile)

    if not args.overlap:
        model.hparams.step = model.hparams.window

    if size > 1:
        dataset = LazySeqDataset(difile=args.difile, hparams=argparse.Namespace(**model.hparams),
                                 keep_open=True, comm=comm, size=size, rank=rank, bal_gnm=True)
    else:
        dataset = LazySeqDataset(difile=args.difile, hparams=argparse.Namespace(**model.hparams), keep_open=True)

    tot_bases = dataset.difile.get_seq_lengths().sum()
    args.logger.info(f'rank {rank} - processing {tot_bases} bases across {len(dataset)} samples')

    tmp_dset = dataset

    kwargs = dict(batch_size=args.batch_size, shuffle=False)
    if args.num_workers > 0:
        kwargs['num_workers'] = args.num_workers
        kwargs['multiprocessing_context'] = 'spawn'
        #kwargs['worker_init_fn'] = dataset.worker_init
        kwargs['persistent_workers'] = True
    loader = get_loader(tmp_dset, inference=True, **kwargs)

    # return the model, any arguments, and Lighting Trainer args just in case
    # we want to use them down the line when we figure out how to use Lightning for
    # inference

    if not args.features:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    model.eval()

    ret = [model, dataset, loader, args, env, tt]

    if size > 1:
        args.device = torch.device('cuda:%d' % local_rank)
    else:
        args.device = torch.device('cuda')

    return tuple(ret)


def run_inference(argv=None):
    """Run inference using PyTorch

    Args:
        argv: a command-line string or argparse.Namespace object to use for running inference
              If none are given, read from command-line i.e. like running argparse.ArgumentParser.parse_args
    """

    args = parse_args(argv=argv)
    model, dataset, loader, args, env, tt = process_args(args)

    f_kwargs = dict()
    if env is not None:
        rank = env.global_rank()
        size = env.world_size()

        from mpi4py import MPI
        args.logger.info(f'rank {rank} - Using MPI-IO')
        f_kwargs['driver'] = 'mpio'
        f_kwargs['comm'] = MPI.COMM_WORLD

    parallel_chunked_inf_summ(model, dataset, loader, args, f_kwargs, tt)


def _compute_taxonomy_transforms(tt):
    tt = tt.copy()
    tt['species'] = np.arange(len(tt))
    transforms = list()
    levels = tt.columns[1:].values
    for i in range(1, len(levels))[::-1]:
        lower = tt[levels[i]].values
        upper = tt[levels[i-1]].values
        mat = np.zeros((lower.max() + 1, upper.max() + 1), dtype=np.float32)
        mat[lower, upper] = 1.0
        transforms.append(torch.from_numpy(mat))
    return transforms, levels.astype(np.string_)



### BEGIN: Helper functions for parallel_chunked_inf_summ
def _write_hier_maxprobs(idx, outputs_q, n_levels, maxprob, tt, maxprobs_dset, preds_dset):
    maxprobs = np.zeros((len(idx), n_levels, maxprob)) - 1.0
    preds = np.zeros((len(idx), n_levels), dtype=int)
    n_highest_taxa = tt[:, 0].max() + 1
    # for each taxonomic level in each sequence, compute the
    # max l probabilities normalized across probabilities
    # for taxa within the upper taxonomic level's classification
    # e.g. if a sequence is classified as an archaea, limit phylum
    # classification to archaeal phyla
    for seq_i in range(len(idx)):
        mask = np.arange(n_highest_taxa)
        for lvl_i in range(len(outputs_q)):
            # get probabilities for taxa that are in the taxa classified
            # in the upper level
            valid_probs = outputs_q[lvl_i][seq_i][mask]
            valid_probs /= valid_probs.sum()

            k = max(valid_probs.shape[0] - maxprob, 0)
            l = valid_probs.shape[0] - k
            maxprobs[seq_i, lvl_i, :l] = np.sort(np.partition(valid_probs, k)[k:])[::-1]

            pred = mask[np.argmax(valid_probs)]
            if lvl_i + 1 < tt.shape[1]:
                mask = np.unique(tt[tt[:, lvl_i] == pred, lvl_i + 1])
            preds[seq_i, lvl_i] = pred

    maxprobs_dset[idx] = maxprobs
    preds_dset[idx] = preds

def _write_maxprobs(idx, outputs_q, n_levels, maxprob, maxprobs_dset):
    maxprobs = np.zeros((len(idx), n_levels, maxprob)) - 1.0
    # for each taxonomic level in each sequence, compute the
    # max l probabilities
    for seq_i in range(len(idx)):
        for lvl_i in range(len(outputs_q)):
            oq = outputs_q[lvl_i][seq_i]
            k = max(oq.shape[0] - maxprob, 0)
            l = oq.shape[0] - k
            maxprobs[seq_i, lvl_i, :l] = np.sort(np.partition(oq, k)[k:])[::-1]
    maxprobs_dset[idx] = maxprobs


def _write_preds(idx, outputs_q, n_levels, maxprob, preds_dset):
    preds = np.zeros((len(idx), n_levels), dtype=int)
    # for each taxonomic level in each sequence, compute the
    # prediction based on the maximum probability
    for seq_i in range(len(idx)):
        for lvl_i in range(len(outputs_q)):
            preds[seq_i, lvl_i] = np.argmax(outputs_q[lvl_i][seq_i])
    preds_dset[idx] = preds


def _compute_mean(idx, outputs_q, counts_q):
    for oq in outputs_q:
        for i in range(len(idx)):
            oq[i] /= counts_q[i]

def _reduce_chunks(labels_tt, ids, outputs, chunk_labels):
    """ Reduce outputs and labels for each item

    Args:
        labels_tt:  the integerized taxonomy table
        ids:        the ids for each chunk
        outputs:    the output for each chunk

    Returns:
        items:          the unique items from *ids*
        counts:         the number of occurence of each unique item in *ids*
        outputs_sum:    the reduced output for each unique item
        labels:         the label for each item
    """
    values, counts = np.unique(ids, return_counts=True)
    outputs_sum = [[] for i in range(len(outputs))]
    labels = list()
    for i in values:
        mask = ids == i
        for o_sum, _output in zip(outputs_sum, outputs):
            o_sum.append(_output[mask].sum(axis=0))
        labels.append(labels_tt[chunk_labels[mask][0]])
    return values, counts, outputs_sum, labels


def _update_queues(items_q, items, outputs_q, outputs_sum, counts_q, counts, labels_q, labels):
    """Update queues
    Args:
        items_q:        the IDs of the items in queue
        items:          the IDs of the items at this iteration
        outputs_q:      the outputs of the items in the queue
        outputs_sum:    the reduced outputs for the chunks of each items at this iteration
        counts_q:       the number of chunks that have gone into each item in the queue
        counts:         the number of chunks for each item at this iteration
        labels_q:       the labels of each item in the queue
        labels:         the labels for each item at this iteration
    """
    # Add the first sum of this iteration to the last sum of
    # the previous iteration if they belong to the same sequence
    if len(items_q) > 0 and items_q[-1] == items[0]:
        for oq, osum in zip(outputs_q, outputs_sum):
            oq[-1] += osum[0]
        counts_q[-1] += counts[0]
        # drop the first sum so we don't end up with duplicates
        items = items[1:]
        counts = counts[1:]
        outputs_sum = [osum[1:] for osum in outputs_sum]
        labels = labels[1:]

    for oq, osum in zip(outputs_q, outputs_sum):
        oq.extend(osum)
    items_q.extend(items)
    counts_q.extend(counts)
    labels_q.extend(labels)

def _flush_queues(args, labels_tt, items_q, outputs_q, counts_q, maxprobs_dset, preds_dset, labels_q, labels_dset, final=False):
    """Empty queues
    Args:
        args:           the args from parallel_chunked_inf_summ
        labels_tt:      the integerized taxonomy table
        items_q:        the IDs of the items in queue
        outputs_q:      the outputs of the items in the queue
        counts_q:       the number of chunks that have gone into each item in the queue
        maxprobs_dset:  the dataset to write maxprobabilities to
        preds_dset:     the dataset to write predictions to
        labels_q:       the labels of each item in the queue
        labels_dset:    the dataset to write labels to
        final:          True if this is the final flush of the queues, False otherwise
    """
    # do not write the last sequence in case
    # there are more chunks left for it
    if final:
        idx = items_q
    else:
        idx = items_q[:-1]


    n_levels = labels_tt.shape[1]

    _compute_mean(idx, outputs_q, counts_q)

    labels_dset[idx] = labels_q if final else labels_q[:-1]

    if args.hierarchy:
        _write_hier_maxprobs(idx, outputs_q, n_levels, args.maxprob, labels_tt, maxprobs_dset, preds_dset)
    else:
        _write_maxprobs(idx, outputs_q, n_levels, args.maxprob, maxprobs_dset)
        _write_preds(idx, outputs_q, n_levels, args.maxprob, preds_dset)

    # drop everything except for the last sequence
    # in case there are more chunks left for it
    for oq in outputs_q:
        del oq[:-1]
    del items_q[:-1]
    del counts_q[:-1]
    del labels_q[:-1]

    return idx
### END: Helper functions for parallel_chunked_inf_summ


def parallel_chunked_inf_summ(model, dataset, loader, args, fkwargs, tt):

    transforms, levels = _compute_taxonomy_transforms(tt)

    n_samples = args.total_seqs
    n_levels = len(levels)
    args.logger.debug(f"rank {dataset.rank} - n_samples = {n_samples}")
    all_seq_ids = dataset.difile.id #get_sequence_subset()
    seq_lengths  = dataset.difile.lengths #get_seq_lengths()
    all_gnm_ids = np.unique(dataset.difile.genomes)

    f = h5py.File(args.output, 'w', **fkwargs)
    outputs_dset = None
    if args.save_chunks:
        outputs_dset = f.require_dataset('outputs', shape=(n_samples, args.n_outputs), dtype=float)

    # Set up datasets for sequences
    seq_grp = f.create_group("sequences")
    args.logger.debug(f"rank {dataset.rank} - making labels")
    labels_dset = seq_grp.require_dataset('labels', shape=(n_samples, n_levels), dtype=int, fillvalue=-1)
    labels_dset.attrs['n_classes'] = args.n_outputs
    args.logger.debug(f"rank {dataset.rank} - making maxprob")
    maxprobs_dset = seq_grp.require_dataset('maxprob', shape=(n_samples, n_levels, args.maxprob), dtype=float)
    args.logger.debug(f"rank {dataset.rank} - making preds")
    preds_dset = seq_grp.require_dataset('preds', shape=(n_samples, n_levels,), dtype=int, fillvalue=-1)
    args.logger.debug(f"rank {dataset.rank} - making lengths")
    seqlen_dset = seq_grp.require_dataset('lengths', shape=(n_samples,), dtype=int, fillvalue=-1)

    if all_seq_ids is None:
        seqlen_dset[:] = seq_lengths
    else:
        args.logger.debug(f"rank {dataset.rank} - writing seq_lengths")
        # iterate over indices individually - passing this off to h5py takes prohibitively long
        for i in range(len(seq_lengths)):
            seqlen_dset[all_seq_ids[i]] = seq_lengths[i]

    gnm_grp = f.create_group("genomes")

    args.logger.debug(f"rank {dataset.rank} - making labels")
    gnm_labels_dset = gnm_grp.require_dataset('labels', shape=(n_samples, n_levels), dtype=int, fillvalue=-1)
    gnm_labels_dset.attrs['n_classes'] = args.n_outputs
    args.logger.debug(f"rank {dataset.rank} - making maxprob")
    gnm_maxprobs_dset = gnm_grp.require_dataset('maxprob', shape=(n_samples, n_levels, args.maxprob), dtype=float)
    args.logger.debug(f"rank {dataset.rank} - making preds")
    gnm_preds_dset = gnm_grp.require_dataset('preds', shape=(n_samples, n_levels,), dtype=int, fillvalue=-1)
    args.logger.debug(f"rank {dataset.rank} - making lengths")
    gnmlen_dset = gnm_grp.require_dataset('lengths', shape=(n_samples,), dtype=int, fillvalue=-1)
    gnm_lengths = np.bincount(dataset.difile.genomes, weights=dataset.difile.lengths)

    if all_seq_ids is None:
        gnmlen_dset[:] = gnm_lengths
    else:
        gnm_lengths = gnm_lengths[all_gnm_ids]
        args.logger.debug(f"rank {dataset.rank} - writing gnm_lengths")
        # iterate over indices individually - passing this off to h5py takes prohibitively long
        for gnm_i, gnm_len in zip(all_gnm_ids, gnm_lengths):
            gnmlen_dset[gnm_i] = gnm_len


    levels_dset = f.create_dataset('levels', shape=(n_levels,), dtype=levels.dtype)
    rank = 0
    if 'comm' in fkwargs:
        fkwargs['comm'].Get_rank()
    if rank == 0:
        levels_dset[:] = levels

    # to-write queues - we use those so we're not doing I/O at every iteration
    outputs_q = [[] for i in range(len(transforms) + 1)]
    gnm_outputs_q = [[] for i in range(len(transforms) + 1)]
    labels_q = list()
    gnm_labels_q = list()
    counts_q = list()
    gnm_counts_q = list()
    seqs_q = list()
    genomes_q = list()

    labels_tt = tt.iloc[:, 1:]
    labels_tt['species'] = np.arange(len(labels_tt))
    labels_tt = labels_tt.values

    # write what's in the to-write queues
    if not hasattr(args, 'n_seqs'):
        args.n_seqs = 500

    # send model to GPU
    args.logger.debug(f"rank {dataset.rank} - sending model to args.device")
    model.to(args.device)

    uniq_labels = set()
    args.logger.debug(f"rank {dataset.rank} - reading and processing")
    go_args = [model, loader, args.device]
    go_kwargs = dict(debug=args.debug, chunks=args.n_batches, prog_bar=dataset.rank==0, transforms=transforms)

    for idx, _outputs, _labels, _orig_lens, _seq_ids, _genome_ids in get_outputs(*go_args, **go_kwargs):

        uniq_labels.update(_labels)

        # sum the outputs of the chunks for each sequence
        seqs, counts, outputs_sum, labels = _reduce_chunks(labels_tt, _seq_ids, _outputs, _labels)

        # sum the outputs of the chunks for each genome
        genomes, gnm_counts, gnm_outputs_sum, gnm_labels = _reduce_chunks(labels_tt, _genome_ids, _outputs, _labels)

        # update the queues for each sequence
        _update_queues(seqs_q, seqs, outputs_q, outputs_sum, counts_q, counts, labels_q, labels)

        # update the queues for each genome
        _update_queues(genomes_q, genomes, gnm_outputs_q, gnm_outputs_sum, gnm_counts_q, gnm_counts, gnm_labels_q, gnm_labels)

        # flush the sequences queues once we have more than specified
        if len(seqs_q) > args.n_seqs:
            idx = _flush_queues(args, labels_tt, seqs_q,
                                outputs_q, counts_q, maxprobs_dset, preds_dset,
                                labels_q, labels_dset)
            args.logger.debug(f"rank {dataset.rank} - saving these sequences {idx}")

        # flush the genomes queues once we have more than specified
        if len(genomes_q) > args.n_seqs:
            idx = _flush_queues(args, labels_tt, genomes_q,
                                gnm_outputs_q, gnm_counts_q, gnm_maxprobs_dset, gnm_preds_dset,
                                gnm_labels_q, gnm_labels_dset)
            args.logger.debug(f"rank {dataset.rank} - saving these genomes {idx}")

    args.logger.debug(f"rank {dataset.rank} - came across these labels {list(sorted(uniq_labels))}")
    # clean up what's left in the to-write queue

    idx = _flush_queues(args, labels_tt, seqs_q,
                        outputs_q, counts_q, maxprobs_dset, preds_dset,
                        labels_q, labels_dset, final=True)
    args.logger.debug(f"rank {dataset.rank} - saving these sequences {idx}")

    idx = _flush_queues(args, labels_tt, genomes_q,
                        gnm_outputs_q, gnm_counts_q, gnm_maxprobs_dset, gnm_preds_dset,
                        gnm_labels_q, gnm_labels_dset, final=True)
    args.logger.debug(f"rank {dataset.rank} - saving these genomes {idx}")

    args.logger.info(f'rank {dataset.rank} - closing {args.output}')
    f.close()


def get_outputs(model, loader, device, debug=False, chunks=None, prog_bar=True, transforms=None):
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
    max_batches = 100 if debug else sys.maxsize
    from tqdm import tqdm
    file = sys.stdout
    it = loader
    if prog_bar:
        it = tqdm(it, file=sys.stdout)

    n_levels = 1
    if transforms is not None:
        transforms = [v.to(device) for v in transforms]
        n_levels = len(transforms) + 1

    indices, outputs, labels, orig_lens, seq_ids, genomes = [], [[] for i in range(n_levels)], [], [], [], []

    with torch.no_grad():
        for idx, (i, X, y, olen, seq_i, genome) in enumerate(it):
            X = X.to(device)
            with autocast():
                out = model(X)

            if transforms is not None:
                all_outputs = [out]
                for tfm in transforms:
                    all_outputs.append(all_outputs[-1].matmul(tfm))
                    all_outputs[-2] = all_outputs[-2].to('cpu').detach()
                all_outputs[-1] = all_outputs[-1].to('cpu').detach()
                out = all_outputs[::-1]
            else:
                out = [out.to('cpu').detach()]

            for o_i in range(n_levels):
                outputs[o_i].append(out[o_i])

            indices.append(i.to('cpu').detach())
            labels.append(y.to('cpu').detach())
            orig_lens.append(olen.to('cpu').detach())
            seq_ids.append(seq_i.to('cpu').detach())
            genomes.append(genome.to('cpu').detach())
            if idx >= max_batches:
                break
            if chunks and (idx+1) % chunks == 0:
                yield cat(indices, outputs, labels, orig_lens, seq_ids, genomes)
                indices, outputs, labels, orig_lens, seq_ids, genomes = [], [[] for i in range(n_levels)], [], [], [], []
    if chunks is None:
        return cat(indices, outputs, labels, orig_lens, seq_ids, genomes)
    else:
        if len(indices) > 0:
            yield cat(indices, outputs, labels, orig_lens, seq_ids, genomes)


def cat(indices, outputs, labels, orig_lens, seq_ids, genome):
    ret = (torch.cat(indices).numpy(),
           [torch.cat(o).numpy() for o in outputs],
           torch.cat(labels).numpy(),
           torch.cat(orig_lens).numpy(),
           torch.cat(seq_ids).numpy(),
           torch.cat(genome).numpy())
    return ret


from . import models  # noqa: E402

if __name__ == '__main__':
    run_inference()
