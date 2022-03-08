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

def build_deployment_pkg(argv=None):
    """
    Convert a Torch model checkpoint to ONNX format
    """

    import json
    import os
    import shutil
    import tempfile
    import zipfile
    from hdmf.common import get_hdf5io

    desc = "Convert a Torch model checkpoint to ONNX format"
    epi = ("By default, the ONNX file will be written to same directory "
           "as checkpoint")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('input', type=str, help='the input file to run inference on')
    parser.add_argument('config', type=str, help='the config file used for training')
    parser.add_argument('nn_model', type=str, help='the NN model for doing predictions')
    parser.add_argument('conf_model', type=str, help='the checkpoint file to use for running inference')
    parser.add_argument('output_dir', type=str, help='the directory to copy to before zipping')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='overwrite output if it exists')
    parser.add_argument('-s', '--softmax', action='store_true', default=False, help='add softmax layer to model before exporting')

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    logger = get_logger()


    if os.path.exists(args.output_dir):
        print(f"{args.output_dir} exists, exiting")
        exit(1)

    os.mkdir(args.output_dir)
    tmpdir = args.output_dir

    logger.info(f'Using temporary directory {tmpdir}')
    logger.info(f'loading sample input from {args.input}')

    io = get_hdf5io(args.input, 'r')
    difile = io.read()
    tt = difile.taxa_table
    _load = lambda x: x[:]
    for col in tt.columns:
        col.transform(_load)
    tt_df = tt.to_dataframe().set_index('taxon_id')
    io.close()

    path = lambda x: os.path.join(tmpdir, os.path.basename(x))

    manifest = {
        'taxa_table': os.path.join(tmpdir, "taxa_table.csv"),
        'nn_model': path(args.nn_model),
        'conf_model': path(args.conf_model),
        'training_config': path(args.config)
    }

    logger.info(f"exporting taxa table CSV to {manifest['taxa_table']}")
    tt_df.to_csv(manifest['taxa_table'])
    logger.info(f"copying {args.nn_model} to {manifest['nn_model']}")
    shutil.copyfile(args.nn_model, manifest['nn_model'])
    logger.info(f"copying {args.conf_model} to {manifest['conf_model']}")
    shutil.copyfile(args.conf_model, manifest['conf_model'])
    logger.info(f"copying {args.config} to {manifest['training_config']}")
    shutil.copyfile(args.config, manifest['training_config'])

    with open(os.path.join(tmpdir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)


    zip_path = args.output_dir + ".zip"
    zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(tmpdir):
        for file in files:
            path = os.path.join(root, file)
            logger.info(f'adding {path} to {zip_path}')
            zipf.write(path)

    zipf.close()

    logger.info(f'removing {tmpdir}')
    shutil.rmtree(tmpdir)
