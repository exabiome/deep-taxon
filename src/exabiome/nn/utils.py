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


def process_model(args, inference=False):
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
            hparams = torch.load(args.checkpoint, map_location=torch.device('cpu'))['hyper_parameters']
            model = model_cls.load_from_checkpoint(args.checkpoint, hparams=hparams)
        except RuntimeError as e:
            if 'Missing key(s)' in e.args[0]:
                raise RuntimeError(f'Unable to load checkpoint. Make sure {args.checkpoint} is a checkpoint for {args.model}') from e
            else:
                raise e

    # if we pretrained a feature extracter, like a resnet*_feat model
    # load the pretrained model and add a classifier
    #elif args.features_checkpoint is not None:
    #    feat_model_hparams = torch.load(args.features, map_location=torch.device('cpu'))['hyper_parameters']
    #    feat_model_name = feat_model_hparams['model']
    #    from .models.ssl import ResNetClassifier
    #    if not (feat_model_name.startswith('resnet') and feat_model_name.endswith('feat')):
    #        raise ValueError("Cannot add classifier to model %s -- must be a ResNet feature model (i.e. resnet*_feat)" % feat_model_name)

    #    # load the features model
    #    features_model = models._models[feat_model_name].load_from_checkpoint(args.features, hparams=feat_model_hparams)
    #    args.feat_model_hparams = feat_model_hparams
    #    args.feat_model_name = feat_model_name
    #    _check_hparams(args)
    #    model = model_cls(args, features=features_model)
    # elif args.features is not None:
    #     feat_model_hparams = torch.load(args.features, map_location=torch.device('cpu'))['hyper_parameters']
    #     feat_model_name = feat_model_hparams['model']
    #     from .models.ssl import ResNetClassifier
    #     if not (feat_model_name.startswith('resnet') and feat_model_name.endswith('feat')):
    #         raise ValueError("Cannot add classifier to model %s -- must be a ResNet feature model (i.e. resnet*_feat)" % feat_model_name)

    #     # load the features model
    #     features_model = models._models[feat_model_name].load_from_checkpoint(args.features, hparams=feat_model_hparams)
    #     args.feat_model_hparams = feat_model_hparams
    #     _check_hparams(args)
    #     model = model_cls(args, features=features_model)
    else:
        if not hasattr(args, 'classify'):
            raise ValueError('Parser must check for classify/regression/manifold '
                             'to determine the number of outputs')
        # First, get the dataset, so we can figure
        # out how many outputs there are
        dataset, io = read_dataset(args.input)

        if args.classify:
            n_outputs = len(dataset.difile.taxa_table)
        elif args.manifold:
            n_outputs = args.n_outputs        #TODO make this configurable #breakpoint
        else:
            args.regression = True
            n_outputs = dataset.difile.n_emb_components
        args.n_outputs = n_outputs

        _check_hparams(args)

        model = model_cls(args)

        io.close()

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
