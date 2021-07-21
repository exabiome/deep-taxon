from argparse import Namespace
import sys
import os

from .loader import read_dataset
from . import models

from ..utils import check_directory
from pytorch_lightning.core.decorators import auto_move_data
import torch


def show_models(argv=None):
    '''Summarize what Torch sees in CUDA land'''
    def get_desc(m):
        if model.__doc__ is None:
            return "no description"
        else:
            return model.__doc__.strip().split('\n')[0]
    maxlen = max(list(map(len, models._models.keys())))
    maxlen += (maxlen % 4) + 4

    def pad(s):
        return s + ' '*(maxlen - len(s))

    for name, model in models._models.items():
        print(pad(name), get_desc(model))


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


def _check_hparams(args):
    if args.hparams is not None:
        for k, v in args.hparams.items():
            setattr(args, k, v)
    del args.hparams


def process_model(args, inference=False, taxa_table=None):
    """
    Process a model argument

    Args:
        args (Namespace):       command-line arguments passed by parser
        inference (bool):       load data for inference
    """

    # Next, build our model object so we can get
    # the parameters used if we were given a checkpoint
    model_cls = models._models[args.model]
    if inference:
        model_cls.forward = auto_move_data(model_cls.forward)

    if args.checkpoint is not None:
        try:
            model = model_cls.load_from_checkpoint(args.checkpoint)
            ckpt_hparams = model.hparams
            if not inference:
                if ckpt_hparams.tgt_tax_lvl != args.tgt_tax_lvl:
                    if taxa_table is None:
                        msg = ("Model checkpoint has different taxonomic level than requested -- got {args.tgt_tax_lvl} "
                              "in args, but found {ckpt_hparams.tgt_tax_lvl} in {args.checkpoint}. You must provide the TaxaTable for "
                              "computing the taxonomy mapping for reconfiguring the final output layer")

                        raise ValueError(msg)
                    outputs_map = taxa_table.get_outputs_map(ckpt_hparams.tgt_tax_lvl, args.tgt_tax_lvl)
                    model.reconfigure_outputs(outputs_map)
                    model.hparams.tgt_tax_lvl = args.tgt_tax_lvl
        except RuntimeError as e:
            if 'Missing key(s)' in e.args[0]:
                raise RuntimeError(f'Unable to load checkpoint. Make sure {args.checkpoint} is a checkpoint for {args.model}') from e
            else:
                raise e
    else:
        if not hasattr(args, 'classify'):
            raise ValueError('Parser must check for classify/regression/manifold '
                             'to determine the number of outputs')
        _check_hparams(args)
        #if taxa_table is not None:
        #    args.labels = taxa_table['phylum'].elements.data[:]
        model = model_cls(args)

    return model


def process_output(args, subdir='training_results'):
    """
    Process dataset arguments
    """
    outbase = args.output
    #if args.experiment:
    #    #outbase = os.path.join(outbase, subdir, args.experiment)
    #    outbase = os.path.join(outbase, args.experiment)

    def output(fname):
        return os.path.join(outbase, fname)

    return outbase, output
