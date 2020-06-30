from .train import parse_args
import os
import sys
import copy

import pytorch_lightning as pl
from pytorch_lightning import Callback

import optuna
from optuna.integration import PyTorchLightningPruningCallback

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Objective:
    """
    A helper class for building the objective function
    """

    def __init__(self, model_cls, hparams, dataset, monitor_metric='val_loss', targs=None):
        self.model_cls = model_cls
        self.hparams = hparams
        self.dataset = dataset
        self.monitor_metric = monitor_metric
        if targs is not None:
            # do not let user overwrite checkpoint_callback or callbacks
            if 'checkpoint_callback' in targs:
                raise ValueError("checkpoint_callback cannot be specified if running with optuna")
            if 'early_stop_callback' in targs:
                raise ValueError("early_stop_callback cannot be specified if running with optuna")
            if 'callbacks' in targs:
                if isinstance(targs['callbacks'], (tuple, list)):
                    targs['callbacks'] = list(targs['callbacks']) + [metrics_callback]
                elif isinstance(targs['callbacks'], Callback):
                    targs['callbacks'] = [targs['callbacks'], metrics_callback]
                else:
                    raise ValueError("callbacks must be a Callback, a tuple, or a list")
        else:
            targs = dict()
        self.targs = targs

    def get_hparams(self, trial):
        ret = dict()
        ret['dropout_rate'] = trial.suggest_uniform('dropout_rate', 0.0, 1.0)
        return ret

    def __call__(self, trial):
        # Filenames for each trial must be made unique in order to access each checkpoint.
        ckpt_path = os.path.join(self.hparams.output,
                                 trial.study.study_name,
                                 "trial_{}".format(trial.number),
                                 "{epoch:03d}")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(ckpt_path, monitor=self.monitor_metric)

        # set hyperparameters under optimization
        hparams = copy.copy(self.hparams)
        for k, v in self.get_hparams(trial).items():
            setattr(hparams, k, v)
        model = self.model_cls(hparams)
        model.set_dataset(self.dataset)

        # The default logger in PyTorch Lightning writes to event files to be consumed by
        # TensorBoard. We don't use any logger here as it requires us to implement several abstract
        # methods. Instead we setup a simple callback, that saves metrics from each validation step.
        metrics_callback = MetricsCallback()

        # set up arguments required for integrating with optuna
        _targs = dict(
            logger=False,
            checkpoint_callback=checkpoint_callback,
            callbacks=[metrics_callback],
            early_stop_callback=PyTorchLightningPruningCallback(trial, monitor=self.monitor_metric),
        )
        _targs.update(self.targs)

        trainer = pl.Trainer(**_targs)
        trainer.fit(model)

        return metrics_callback.metrics[-1][self.monitor_metric]


def parse_args(*addl_args, argv=None, return_io=False):
    """
    Parse arguments for training executable
    """
    import argparse
    from ..utils import parse_seed

    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, str):
        argv = argv.strip().split()

    desc = "run network hyperparameter optimization"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('storage', help='Optuna storage database')
    parser.add_argument('study_name', help='the study to run trials for')
    parser.add_argument('output', type=str, help='file to checkpoints to')
    parser.add_argument('-t', '--n_trials', type=int, default=1, help='the number of trials to run')
    parser.add_argument('-s', '--seed', type=parse_seed, default='', help='seed to use for train-test split in this study')


    group = parser.add_argument_group(title='Training runtime arguments',
                                      description='Arguments that configure how training is run')
    group.add_argument('-g', '--gpus', nargs='?', const=True, default=False, help='use GPU')
    group.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('-L', '--load', action='store_true', default=False, help='load data into memory before running training loop')

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    study = optuna.create_study(storage=args.storage,
                                study_name=args.study_name,
                                load_if_exists=True)

    # set seed of study if it has not yet been set
    # otherwise, ignore the provided seed
    if hasattr(study, 'seed'):
        args.seed = study.seed
    else:
        study.set_user_attr('seed', args.seed)

    n_trials = args.n_trials

    del args.storage
    del args.study_name
    del args.n_trials

    return study, n_trials, args


def optuna_run(train_cmdline, objective_cls=Objective):
    from .train import parse_args as parse_train_args, process_args as process_train_args

    study, n_trials, rt_args = parse_args()

    train_args = parse_train_args(argv=train_cmdline)

    for k,v in vars(rt_args).items():
        setattr(train_args, k, v)

    model_cls, dataset, hparams, addl_targs = process_train_args(train_args)

    objective = objective_cls(model_cls, hparams, dataset, targs=addl_targs)

    study.optimize(objective, n_trials=n_trials)
