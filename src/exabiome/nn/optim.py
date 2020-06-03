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


def _get_objective_helper(model=None, monitor_metric='val_loss', targs=None):
    """
    A helper functions for building the objective function
    """
    def objective(trial):
        # Filenames for each trial must be made unique in order to access each checkpoint.
        ckpt_path = os.path.join('optuna',
                                 trial.study.study_name,
                                 "trial_{}".format(trial.number),
                                 "{epoch:03d}")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(ckpt_path, monitor=monitor_metric)

        # The default logger in PyTorch Lightning writes to event files to be consumed by
        # TensorBoard. We don't use any logger here as it requires us to implement several abstract
        # methods. Instead we setup a simple callback, that saves metrics from each validation step.
        metrics_callback = MetricsCallback()

        # set up arguments required for integrating with optuna
        _targs = dict(
            logger=False,
            checkpoint_callback=checkpoint_callback,
            callbacks=[metrics_callback],
            early_stop_callback=PyTorchLightningPruningCallback(trial, monitor=monitor_metric),
        )

        if targs is not None:
            targs = targs.copy()
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
            _targs.update(targs)


        trainer = pl.Trainer(**_targs)

        trainer.fit(model)

        return metrics_callback.metrics[-1][monitor_metric]

    return objective

def get_objective(cmdline):
    """
    Return an Optuna objective for the given command-line arguments
    """

    model_cls, dataset, args, addl_targs = parse_args(argv=cmdline)

    model = model_cls(args)
    model.set_dataset(dataset)

    return _get_objective_helper(model, targs=addl_targs)
