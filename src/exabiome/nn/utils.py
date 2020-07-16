import sys
import os

from .loader import read_dataset
from . import models

from ..utils import check_directory
from pytorch_lightning.core.decorators import auto_move_data


def process_gpus(gpus):
    ret = gpus
    if isinstance(ret, str):
        ret = [int(g) for g in ret.split(',')]
        if len(ret) == 1:
            # user specified the number of GPUs to use,
            # else assume they just specified which specific GPUS to use
            ret = ret[0]
    elif isinstance(ret, bool):
        if ret:
            # use all GPUs
            ret = -1
        else:
            # don't use any GPUs
            ret = None
    return ret


def process_model(args, inference=False):
    """
    Process a model argument

    Args:
        args (Namespace):       command-line arguments passed by parser
        inference (bool):       load data for inference
    """
    # First, get the dataset, so we can figure
    # out how many outputs there are
    dataset, io = read_dataset(args.input)

    # Next, build our model object so we can get
    # the parameters used if we were given a checkpoint
    model = models._models[args.model]
    if inference:
        model.forward = auto_move_data(model.forward)
    del args.model

    if args.checkpoint is not None:
        model = model.load_from_checkpoint(args.checkpoint)
    else:
        if not hasattr(args, 'classify'):
            raise ValueError('Parser must check for classify/regression/manifold '
                             'to determine the number of outputs')
        if args.classify:
            n_outputs = len(dataset.difile.taxa_table)
        elif args.manifold:
            n_outputs = 32        #TODO make this configurable #breakpoint
        else:
            args.regression = True
            n_outputs = dataset.difile.n_emb_components
        args.n_outputs = n_outputs

        if args.hparams is not None:
            for k, v in args.hparams.items():
                setattr(args, k, v)
        del args.hparams

        model = model(args)

    io.close()

    return model


def process_output(args, subdir='training_results'):
    """
    Process dataset arguments
    """
    outbase = args.output
    if args.experiment:
        outbase = os.path.join(outbase, subdir, args.experiment)
    check_directory(outbase)

    def output(fname):
        return os.path.join(outbase, fname)

    return outbase, output
