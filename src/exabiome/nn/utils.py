import sys
import os

from exabiome.sequence import AbstractChunkedDIFile, WindowChunkedDIFile
from . import SeqDataset, train_test_loaders
from hdmf.common import get_hdf5io

from . import models

from ..utils import check_directory


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


def check_window(window, step):
    if window is None:
        return None, None
    else:
        if step is None:
            step = window
        return window, step


def read_dataset(path):
    hdmfio = get_hdf5io(path, 'r')
    difile = hdmfio.read()
    dataset = SeqDataset(difile)
    return dataset, hdmfio


def process_model_and_dataset(args, inference=False):
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
    del args.model

    if args.checkpoint is not None:
        model = model.load_from_checkpoint(args.checkpoint)
        args.window = model.hparams.window
        args.step = model.hparams.step
    else:
        if not hasattr(args, 'classify'):
            raise ValueError('Parser must check for classify/regression/manifold '
                             'to determine the number of outputs')
        if args.classify:
            dataset.set_classify(True)
            n_outputs = len(dataset.difile.taxa_table)
        elif args.manifold:
            dataset.set_classify(True)
            n_outputs = 32        #TODO make this configurable #breakpoint
        else:
            args.regression = True
            dataset.set_classify(False)
            n_outputs = dataset.difile.n_emb_components
        args.n_outputs = n_outputs

        model = model(args)
        args.window, args.step = check_window(args.window, args.step)

    # Process any arguments that impact how we set up the dataset
    if args.window is not None:
        dataset.difile = WindowChunkedDIFile(dataset.difile, args.window, args.step)
    if args.load:
        dataset.load()

    # Finally, set the dataset on the model
    model.set_dataset(dataset, inference=inference)

    return model, dataset, io


def process_output(args):
    """
    Process dataset arguments
    """
    outbase = args.output
    if args.experiment:
        outbase = os.path.join(outbase, 'training_results', args.experiment)
    check_directory(outbase)

    def output(fname):
        return os.path.join(outbase, fname)

    return outbase, output
