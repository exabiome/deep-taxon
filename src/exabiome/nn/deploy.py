import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.onnx

from ..utils import get_logger
from .loader import LazySeqDataset
from .train import process_config
from .utils import process_model

def process_args(argv=None, size=1, rank=0, comm=None):
    """
    Process arguments for running inference
    """
    if not isinstance(argv, argparse.Namespace):
        args = parse_args(argv=argv)
    else:
        args = argv

class SoftmaxModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return self.sm(x)

def to_onnx(argv=None):
    """
    Convert a Torch model checkpoint to ONNX format
    """

    desc = "Convert a Torch model checkpoint to ONNX format"
    epi = ("By default, the ONNX file will be written to same directory "
           "as checkpoint")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('config', type=str, help='the config file used for training')
    parser.add_argument('input', type=str, help='the input file to run inference on')
    parser.add_argument('checkpoint', type=str, help='the checkpoint file to use for running inference')
    parser.add_argument('-o', '--output', type=str, help='the file to save outputs to', default=None)
    parser.add_argument('-f', '--force', action='store_true', default=False, help='overwrite output if it exists')
    parser.add_argument('-s', '--softmax', action='store_true', default=False, help='add softmax layer to model before exporting')

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    logger = get_logger()

    conf_args = process_config(args.config)
    for k, v in vars(conf_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if args.output is None:
        args.output = f'{os.path.splitext(args.checkpoint)[0]}.onnx'

    if os.path.exists(args.output) and not args.force:
        print(f'ONNX file {args.output} already exists. Use -f to overwrite', file=sys.stderr)
        sys.exit(1)



    logger.info(f'loading sample input from {args.input}')
    dataset = LazySeqDataset(path=args.input, hparams=args, keep_open=True)
    input_sample = torch.stack([dataset[i][1] for i in range(16)])

    # load the model and override batch size
    logger.info(f'loading model from {args.input} using config {args.config}')
    model = process_model(args, inference=True, taxa_table=dataset.difile.taxa_table)

    if args.softmax:
        logger.info(f'adding softmax to {model.__class__.__name__} model')
        model = SoftmaxModel(model)

    model.eval()

    logger.info(f'checking input sample: shape = {input_sample.shape}')
    output = model(input_sample)
    logger.info(f'output shape = {output.shape}')


    logger.info(f'writing ONNX file to {args.output}')
    with torch.no_grad():
        torch.onnx.export(model,                                            # model being run
                          input_sample,                                     # model input (or a tuple for multiple inputs)
                          args.output,                                      # where to save the model (can be a file or file-like object)
                          export_params=True,                               # store the trained parameter weights inside the model file
                          opset_version=10,                                 # the ONNX version to export the model to
                          do_constant_folding=True,                         # whether to execute constant folding for optimization
                          input_names = ['input'],                          # the model's input names
                          output_names = ['output'],                        # the model's output names
                          dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                        'output' : {0 : 'batch_size'}})
