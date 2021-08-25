import numpy as np
import sys
from datetime import datetime
from importlib import import_module

class Command:
    def __init__(self, module, doc):
        ar = f'exabiome.{module}'.split('.')
        self.pkg = '.'.join(ar[:-1])
        self.func = ar[-1]
        self.doc = doc

    def get_func(self):
        return getattr(import_module(self.pkg), self.func)


command_dict = {
    'ncbi-path': Command('gtdb.download.ncbi_path', 'Print path at NCBI FTP site to stdout'),
    'ncbi-fetch': Command('gtdb.download.ncbi_fetch', 'Retrieve sequence data from NCBI FTP site using rsync'),
    'make-fof': Command('gtdb.make_fof.make_fof', 'Find files and print paths for accessions'),
    'prepare-data': Command('gtdb.prepare_data.prepare_data', 'Aggregate sequence data GTDB using a file-of-files'),
    'count-sequence': Command('gtdb.prepare_data.count_sequence', 'Count the length of total sequence length for a set of accessions'),
    'sample-nonrep': Command('gtdb.sample.sample_nonrep', 'Get test strain genomes'),
    'sample-gtdb': Command('gtdb.sample.sample_tree', 'Sample taxa from a tree'),
    'train': Command('nn.train.run_lightning', 'Run training with PyTorch Lightning'),
    'train-conf': Command('nn.train.print_config_options', 'Print the available options for a config file'),
    'conf-tmpl': Command('nn.train.print_config_templ', 'Print an empty config file'),
    'show-args': Command('nn.train.print_args', 'display input arguments for training run'),
    'lr-find': Command('nn.train.lightning_lr_find', 'Run Lightning Learning Rate finder'),
    'cuda-sum': Command('nn.train.cuda_sum', 'Summarize what Torch sees in CUDA land'),
    'extract-loss': Command('nn.extract.extract', 'Exract loss plots from TensorBoard event file'),
    'infer': Command('nn.infer.run_inference', 'Run inference using PyTorch'),
    'summarize': Command('nn.summarize.summarize', 'Summarize training/inference results'),
    'clf-sum': Command('nn.summarize.classifier_summarize', 'Summarize training/inference results'),
    'show-models': Command('nn.utils.show_models', 'Show available models'),
    'train-job': Command('run.run_job.run_train', 'Run a training job'),
    'test-dist': Command('run.disttest.test_dist', 'Broadcast a tensor to test the system'),
    'probe': Command('nn.probe.probe', 'Probe the environment of the system'),
    'test-input': Command('testing.dataset.check_sequences', 'Test input file against original fasta files'),
    'dset-info': Command('nn.loader.dataset_stats', 'Read a dataset and print the number of samples to stdout'),
    'extract-profile': Command('nn.summarize.get_profile_data', 'Extract profile data from log files'),
    'plot-loss': Command('nn.summarize.plot_loss', 'Plot loss curves from metrics file'),
    'show-hparams': Command('nn.train.show_hparams', 'Print hparams from a checkpoint file'),
    'to-onnx': Command('nn.deploy.to_onnx', 'convert checkpoint to ONNX format'),
    'agg-chunks': Command('nn.summarize.aggregate_chunks', 'aggregate sequence chunks to get NN outputs for individual sequences'),
    'agg-seqs': Command('nn.summarize.aggregate_seqs', 'aggregate sequence chunks to get NN outputs for individual sequences'),
    'tax-acc': Command('nn.summarize.taxonomic_accuracy', 'calculate classificaiton accuracy across all possible taxonomic levels'),
}


def print_help():
    print('Usage: deep-index <command> [options]')
    print('Available commands are:\n')
    for c, f in command_dict.items():
        nspaces = 16 - len(c)
        desc = ''
        print(f'    {c}' + ' '*nspaces + f.doc)
    print()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print_help()
    else:
        cmd = sys.argv[1]
        func = command_dict[cmd].get_func()
        func(sys.argv[2:])
