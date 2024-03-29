import argparse
import json
import logging
import os
import os.path as op
import shutil
import sys
import warnings

from hdmf.common import get_hdf5io
import torch
import torch.nn as nn
import torch.onnx
import pytorch_lightning as pl
import numpy as np

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
        args.output = f'{op.splitext(args.checkpoint)[0]}.onnx'

    if op.exists(args.output) and not args.force:
        print(f'ONNX file {args.output} already exists. Use -f to overwrite', file=sys.stderr)
        sys.exit(1)


    io = get_hdf5io(args.input, 'r')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        difile = io.read()

    logger.info(f'loading sample input from {args.input}')
    size = len(difile.seq_table.sequence) // 10000000
    dataset = LazySeqDataset(difile=difile, hparams=args, keep_open=True, size=size, rank=0, load=True)
    input_sample = torch.stack([dataset[i][1] for i in range(16)])

    # load the model and override batch size
    logger.info(f'loading model from {args.input} using config {args.config}')
    model = process_model(args, inference=True, taxa_table=difile.taxa_table)

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


def _compute_taxonomy_transforms(tt):
    tt = tt.copy()
    tt['species'] = np.arange(len(tt))
    transforms = list()
    levels = tt.columns[1:].values
    for i in range(1, len(levels))[::-1]:
        lower = tt[levels[i]].values.astype(int)
        upper = tt[levels[i-1]].values.astype(int)
        mat = np.zeros((lower.max() + 1, upper.max() + 1), dtype=np.float32)
        mat[lower, upper] = 1.0
        transforms.append(torch.from_numpy(mat))
    return transforms, levels.astype(np.string_)


class MultilevelModel(pl.LightningModule):

    def __init__(self, model, transforms):
        super().__init__()
        self.model = model
        self.sm = nn.Softmax(dim=1)
        self.register_buffer('g', transforms[0].to_sparse_csr(), persistent=True)
        self.register_buffer('f', transforms[1].to_sparse_csr(), persistent=True)
        self.register_buffer('o', transforms[2].to_sparse_csr(), persistent=True)
        self.register_buffer('c', transforms[3].to_sparse_csr(), persistent=True)
        self.register_buffer('p', transforms[4].to_sparse_csr(), persistent=True)
        self.register_buffer('d', transforms[5].to_sparse_csr(), persistent=True)
        self.parse = torch.cumsum(torch.Tensor([self.d.shape[1],
                                                self.p.shape[1],
                                                self.c.shape[1],
                                                self.o.shape[1],
                                                self.f.shape[1],
                                                self.g.shape[1],
                                                self.g.shape[0]]), 0).int()
        self.levels = ['domain', 'phylum', 'class', 'order',
                       'family', 'genus', 'species']

    def forward(self, x):
        s = self.sm(self.model(x))
        g = s.matmul(self.g)
        f = g.matmul(self.f)
        o = f.matmul(self.o)
        c = o.matmul(self.c)
        p = c.matmul(self.p)
        d = p.matmul(self.d)
        return torch.cat([d, p, c, o, f, g, s], dim=1)


def _load_conf_model(conf_model_json, tgtdir):
    with open(conf_model_json, 'r') as f:
        conf_data = json.load(f)

    os.mkdir(tgtdir)
    for lvl_dat in conf_data:
        lvl = lvl_dat['level']
        shutil.copy(op.join(op.dirname(conf_model_json), lvl_dat['model']), tgtdir)
        shutil.copy(op.join(op.dirname(conf_model_json), lvl_dat['roc']), tgtdir)
        lvl_dat['model'] = op.join(op.basename(tgtdir), op.basename(lvl_dat['model']))
        lvl_dat['roc'] = op.join(op.basename(tgtdir), op.basename(lvl_dat['roc']))

    return conf_data


def build_deployment_pkg(argv=None):
    """
    Convert a Torch model checkpoint to ONNX format
    """

    import json
    import shutil
    import tempfile
    import zipfile
    from hdmf.common import get_hdf5io
    import ruamel.yaml as yaml

    desc = "Build the deployment package for GTNet"
    epi = """The conf-model command should be run twice before running this. Once for building
          confidence models for contigs and once for building confidence models for bins."""

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('input', type=str, help='an deep-taxon input file containing sample input and taxonomy table')
    parser.add_argument('config', type=str, help='the config file used for training')
    parser.add_argument('checkpoint', type=str, help='the torch checkpoint for doing predictions')
    parser.add_argument('ctg_conf_model', type=str, help='the contigs confidence models JSON')
    parser.add_argument('bin_conf_model', type=str, help='the bins confidence models JSON')
    parser.add_argument('output_dir', type=str, help='the directory to copy to before zipping')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='overwrite output if it exists')

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    logger = get_logger()

    if op.exists(args.output_dir):
        if args.force:
            logger.info(f"{args.output_dir} exists, removing tree")
            shutil.rmtree(args.output_dir)
        else:
            logger.error(f"{args.output_dir} exists, exiting")
            exit(1)

    os.mkdir(args.output_dir)
    tmpdir = args.output_dir
    logger.info(f'Using temporary directory {tmpdir}')
    path = lambda x: op.join(tmpdir, op.basename(x))

    logger.info(f'Loading config file from {args.config}')
    conf_args = process_config(args.config)
    for k, v in vars(conf_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    logger.info(f'Loading sample input from {args.input}')
    io = get_hdf5io(args.input, 'r')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        difile = io.read()


    tt = difile.taxa_table
    _load = lambda x: x[:]
    for col in tt.columns:
        col.transform(_load)

    tt_df = tt.to_dataframe(index=True)
    transforms, levels = _compute_taxonomy_transforms(tt_df)

    size = len(difile.seq_table.sequence) // 10000000
    size = 2
    dataset = LazySeqDataset(difile=difile, hparams=args, keep_open=True)#, size=size, rank=0, load=True)
    input_sample = torch.stack([dataset[i][1] for i in range(16)])

    # load the model and override batch size
    logger.info(f'Loading model from {args.input} using config {args.config}')
    args.input_nc = len(dataset.vocab)
    model = process_model(args, inference=True, taxa_table=difile.taxa_table)

    logger.info(f'Adding softmax layer and higher level transforms to model')
    model = MultilevelModel(model, transforms)

    logger.info(f'Tracing model')
    ts_out = path(op.splitext(args.checkpoint)[0] + '.pt')
    traced = model.to_torchscript(file_path=ts_out, method='script')

    logger.info(f"Loading confidence model info from {args.bin_conf_model} and {args.ctg_conf_model}")
    conf_data = {
        'contigs': _load_conf_model(args.ctg_conf_model, op.join(tmpdir, 'contigs')),
        'bins': _load_conf_model(args.bin_conf_model, op.join(tmpdir, 'bins'))
    }

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    taxa = list()
    for lvl in tt.colnames[1:]:
        taxa.append({'level': lvl})
        if lvl == 'species':
            taxa[-1]['taxa'] = tt[lvl].data[:].tolist()
        else:
            taxa[-1]['taxa'] = tt[lvl].elements.data[:].tolist()

    manifest = {
        'nn_model': op.basename(ts_out),
        'vocabulary': difile.seq_table.sequence.elements.data[:].tolist(),
        'training_config': config,
        'conf_model': conf_data,
        'taxa': taxa,
    }

    io.close()

    wd = op.dirname(tmpdir)
    zipdir = op.basename(tmpdir)

    with open(op.join(tmpdir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)

    ret_wd = os.getcwd()
    os.chdir(wd)

    zip_path = zipdir + ".zip"
    logger.info(f"Writing deployment package to {op.join(wd, zip_path)}")
    zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)


    for root, dirs, files in os.walk(zipdir):
        for file in files:
            path = op.join(root, file)
            logger.info(f'adding {path} to {zip_path}')
            zipf.write(path)

    zipf.close()

    os.chdir(ret_wd)

    #logger.info(f'removing {tmpdir}')
    #shutil.rmtree(tmpdir)
    #logger.info(f'deployment package saved to {tmpdir}.zip')


def run_onnx_inference(argv=None):
    """
    Convert a Torch model checkpoint to ONNX format
    """

    import argparse
    import json
    from time import time

    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view as swv
    import onnxruntime as ort
    import pandas as pd
    import ruamel.yaml as yaml
    import skbio
    from skbio.sequence import DNA

    from deep_taxon.sequence.convert import DNAVocabIterator
    from deep_taxon.utils import get_logger


    desc = "Run predictions using ONNX"
    epi = ("")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('deploy_dir', type=str, help='the directory containing all the data for deployment')
    parser.add_argument('fastas', nargs='+', type=str, help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-c', '--n_chunks', type=int, default=10000, help='the number of sequence chunks to process at a time')
    parser.add_argument('-F', '--fof', action='store_true', default=False, help='a file-of-files was passed in')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output file to save classifications to')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print specific information about sequences')

    args = parser.parse_args(argv)

    if args.fof:
        with open(args.fastas[0], 'r') as f:
            args.fastas = [s.strip() for s in f.readlines()]

    files = ('taxa_table', 'nn_model', 'conf_model', 'training_config')
    # read manifest
    with open(op.join(args.deploy_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    # remap files in deploy_dir to be relative to where we are running
    for key in files:
        manifest[key] = op.join(args.deploy_dir, op.basename(manifest[key]))

    model_path = manifest['nn_model']
    conf_model_path = manifest['conf_model']
    config_path = manifest['training_config']
    tt_path = manifest['taxa_table']
    vocab = manifest['vocabulary']

    logger = get_logger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    logger.info(f'loading model from {model_path}')
    nn_sess = ort.InferenceSession(model_path, sess_options=so)

    logger.info(f'loading confidence model from {conf_model_path}')
    conf_sess = ort.InferenceSession(conf_model_path, sess_options=so)
    n_max_probs = conf_sess.get_inputs()[0].shape[1] - 1

    logger.info(f'loading training config from {config_path}')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f'loading taxonomy table from {tt_path}')
    tt_df = pd.read_csv(tt_path)
    outcols = ['filename', 'ID'] + tt_df.columns.tolist() + ['score']
    outcols.remove('taxon_id')

    assert nn_sess.get_outputs()[0].shape[1] == len(tt_df)

    logger.info(f'found {len(tt_df)} taxa')

    DNAVocabIterator.chars = ''.join(vocab)
    encode = DNAVocabIterator.encode
    padval = DNAVocabIterator.chars.find('N')
    window = config['window']
    step = config['step']
    k = len(tt_df) - n_max_probs

    all_preds = list()
    all_maxprobs = list()
    all_lengths = list()
    all_seqnames = list()
    all_filepaths = list()

    logger.info(f'beginning inference')
    before = time()

    for fasta_path in args.fastas:
        preds = list()
        logger.info(f'loading {fasta_path}')

        aggregated = list()
        n_seqs = 0
        for seq in skbio.read(fasta_path, format='fasta', constructor=DNA, validate=False):
            n_seqs += 1
            all_seqnames.append(seq.metadata['id'])
            all_lengths.append(len(seq))
            enc = encode(seq)
            starts = np.arange(0, enc.shape[0], step)
            ends = np.minimum(starts + window, enc.shape[0])
            batches = np.ones((len(starts), window), dtype=int) * padval
            for i, (s, e) in enumerate(zip(starts, ends)):
                l = e - s
                batches[i][:l] = enc[s:e]
            logger.debug(f'getting outputs for {all_seqnames[-1]}, {len(batches)} chunks, {all_lengths[-1]} bases')
            outputs = np.zeros(len(tt_df), dtype=float)
            for s in range(0, len(batches), args.n_chunks):
                e = s + args.n_chunks
                outputs += nn_sess.run(None, {'input': batches[s:e]})[0].sum(axis=0)
            outputs /= len(batches)
            aggregated.append(outputs)

        # aggregate everything we just pulled from the fasta file
        all_filepaths.extend([fasta_path] * n_seqs)
        aggregated = np.array(aggregated)

        # get prediction and maximum probabilities for confidence scoring
        logger.info('getting max probabilities')
        preds = np.argmax(aggregated, axis=1)
        all_maxprobs.append(np.sort(np.partition(aggregated, k)[:, k:])[::-1])
        all_preds.append(preds)

    # build input matrix for confidence model
    all_lengths = np.array(all_lengths, dtype=np.float32)
    all_maxprobs = np.concatenate(all_maxprobs)
    conf_input = np.concatenate([all_lengths[:, np.newaxis], all_maxprobs], axis=1, dtype=np.float32)

    # get confidence probabilities
    logger.info('calculating confidence probabilities')
    conf = conf_sess.run(None, {'float_input': conf_input})[1][:, 1]

    # build the final output data frame
    logger.info('building final output data frame')
    all_preds = np.concatenate(all_preds)
    output = tt_df.iloc[all_preds].copy()
    output['filename'] = all_filepaths
    output['ID'] = all_seqnames
    output['score'] = conf
    output = output[outcols]

    # write out data
    if args.output is None:
        outf = sys.stdout
    else:
        outf = open(args.output, 'w')
    output.to_csv(outf, index=False)

    after = time()
    logger.info(f'Took {after - before:.1f} seconds')
