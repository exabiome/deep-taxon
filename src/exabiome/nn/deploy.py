import argparse
import os
import sys

import torch

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
        args.output = f'{args.checkpoint.strip(".ckpt")}.onnx'

    if os.path.exists(args.output) and not args.force:
        print(f'ONNX file {args.output} already exists. Use -f to overwrite', file=sys.stderr)
        sys.exit(1)

    logger.info(f'loading model from {args.input} using config {args.config}')

    # load the model and override batch size
    model = process_model(args, inference=True)
    model.eval()

    logger.info(f'loading sample input from {args.input}')
    dataset = LazySeqDataset(path=args.input, hparams=argparse.Namespace(**model.hparams), keep_open=True)
    input_sample = torch.stack([dataset[i][1] for i in range(10)])

    logger.info(f'writing ONNX file to {args.output}')
    model.to_onnx(args.output, input_sample, export_params=True)
