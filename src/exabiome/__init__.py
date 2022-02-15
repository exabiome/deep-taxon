import numpy as np
from importlib import import_module

class Command:
    def __init__(self, module, doc):
        ar = f'exabiome.{module}'.split('.')
        self.pkg = '.'.join(ar[:-1])
        self.func = ar[-1]
        self.doc = doc

    def get_func(self):
        return getattr(import_module(self.pkg), self.func)


def main():

    command_dict = {
        'Data preparation': {
            'ncbi-path': Command('gtdb.download.ncbi_path', 'Print path at NCBI FTP site to stdout'),
            'ncbi-fetch': Command('gtdb.download.ncbi_fetch', 'Retrieve sequence data from NCBI FTP site using rsync'),
            'make-fof': Command('gtdb.make_fof.make_fof', 'Find files and print paths for accessions'),
            'prepare-data': Command('gtdb.prepare_data.prepare_data', 'Aggregate sequence data GTDB using a file-of-files'),
            'merge-meta': Command('gtdb.prepare_data.merge_metadata', 'Merge metadata files from different sources'),
            'count-sequence': Command('gtdb.prepare_data.count_sequence', 'Count the length of total sequence length for a set of accessions'),
            'sample-nonrep': Command('gtdb.sample.sample_nonrep', 'Get test strain genomes'),
            'sample-gtdb': Command('gtdb.sample.sample_tree', 'Sample taxa from a tree'),
        },
        'Training and models': {
            'train': Command('nn.train.run_lightning', 'Run training with PyTorch Lightning'),
            'train-job': Command('run.run_job.run_train', 'Run a training job'),
            'train-conf': Command('nn.train.print_config_options', 'Print the available options for a config file'),
            'conf-tmpl': Command('nn.train.print_config_templ', 'Print an empty config file'),
            'show-args': Command('nn.train.print_args', 'display input arguments for training run'),
            'show-models': Command('nn.utils.show_models', 'Show available models'),
            'lr-find': Command('nn.train.lightning_lr_find', 'Run Lightning Learning Rate finder'),
            'cuda-sum': Command('nn.train.cuda_sum', 'Summarize what Torch sees in CUDA land'),

        },
        'Training troubleshooting': {
            'filter-metrics': Command('nn.train.filter_metrics', 'Filter metrics by validation or training'),
            'plot-loss': Command('nn.summarize.plot_loss', 'Plot loss curves from metrics file'),
            'extract-loss': Command('nn.extract.extract', 'Exract loss plots from TensorBoard event file'),
            'show-hparams': Command('nn.train.show_hparams', 'Print hparams from a checkpoint file'),
            'fix-model': Command('nn.train.fix_model', 'Fix model trained with old embedding'),
            'test-dist': Command('run.disttest.test_dist', 'Broadcast a tensor to test the system'),
            'probe': Command('nn.probe.probe', 'Probe the environment of the system'),
            'model-info': Command('nn.train.get_model_info', 'Construct and print info about model'),
            'to-onnx': Command('nn.deploy.to_onnx', 'Convert checkpoint to ONNX format'),
            'bm-dset': Command('nn.train.benchmark_pass', 'Read dataset and run a few batches for bencharking'),
        },
        'Inference and model assessment': {
            'infer': Command('nn.infer.run_inference', 'Run inference using PyTorch'),
            'infer-job': Command('run.run_infer.run_inference', 'Run a training job'),
            'agg-chunks': Command('nn.summarize.aggregate_chunks', 'aggregate sequence chunks to get NN outputs for individual sequences'),
            'agg-seqs': Command('nn.summarize.aggregate_seqs', 'aggregate sequence chunks to get NN outputs for individual taxons (i.e. labels)'),
            'tax-acc': Command('nn.summarize.taxonomic_accuracy', 'calculate classificaiton accuracy across all possible taxonomic levels'),
            'summarize': Command('nn.summarize.summarize', 'Summarize training/inference results'),
            'clf-sum': Command('nn.summarize.classifier_summarize', 'Summarize training/inference results'),
            'test-input': Command('testing.dataset.check_sequences', 'Test input file against original fasta files'),
            'dset-info': Command('nn.loader.dataset_stats', 'Read a dataset and print the number of samples to stdout'),
            'extract-profile': Command('nn.summarize.get_profile_data', 'Extract profile data from log files'),
        }
    }
    import sys
    if len(sys.argv) == 1:
        print('Usage: deep-index <command> [options]')
        print('Available commands are:\n')
        for g, d in command_dict.items():
            print(f' {g}')
            for c, f in d.items():
                nspaces = 16 - len(c)
                desc = ''
                print(f'    {c}' + ' '*nspaces + f.doc)
            print()
        print()
    else:
        cmd = sys.argv[1]
        for g, d in command_dict.items():
            func = d.get(cmd)
            if func is not None:
                func = func.get_func()
                break
        if func is not None:
            func(sys.argv[2:])
        else:
            print("Unrecognized command: '%s'" % cmd, file=sys.stderr)
