from pytorch_lightning import LightningModule
import torch_optimizer as ptoptim
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
from time import time

from ..loss import ArcMarginProduct, EuclideanMAELoss, HyperbolicMAELoss, CondensedEuclideanMAELoss, CondensedHyperbolicMAELoss
from .. import TIME_OFFSET

class AbstractLit(LightningModule):

    val_loss = 'validation_loss'
    train_loss = 'training_loss'
    val_acc = 'validation_acc'
    train_acc = 'training_acc'
    test_loss = 'test_loss'

    schedules = ('adam', 'cyclic', 'plateau', 'cosine', 'cosinewr', 'step' )

    def __init__(self, hparams, lr=None, distances=None):
        super().__init__()
        #self.hparams = self.check_hparams(hparams)
        self.save_hyperparameters(hparams)
        if self.hparams.manifold:
            if self.hparams.condensed:
                kwargs = dict(dmat=distances, batch_size=hparams.batch_size)
                if self.hparams.hyperbolic:
                    self._loss = CondensedHyperbolicMAELoss(**kwargs)
                else:
                    self._loss = CondensedEuclideanMAELoss(**kwargs)
            else:
                if self.hparams.hyperbolic:
                    self._loss = HyperbolicMAELoss()
                else:
                    self._loss = EuclideanMAELoss()
        elif self.hparams.classify:
            if self.hparams.tgt_tax_lvl == 'all':
                self._loss = HierarchicalLoss(hparams.n_taxa_all)
            else:
                self._loss = nn.CrossEntropyLoss()
        else:
            self._loss =  nn.MSELoss()

        self.arc = lambda x, y: x
        if self.hparams.arc_margin:
            self.arc = ArcMarginProduct(self.hparams.n_outputs, self.hparams.n_classes)

        self.set_inference(False)
        self.lr = lr or getattr(hparams, 'lr', None)
        self.last_time = time()

    def copy_hparams(self, hparams):
        if isinstance(hparams, argparse.Namespace):
            hparams = vars(hparams)
        for k, v in hparams.items():
            if k in self.hparams:
                self.hparams[k] = v

    @staticmethod
    def check_hparams(hparams):
        if isinstance(hparams, dict):
            return argparse.Namespace(**hparams)
        return hparams

    def set_class_weights(self, weights):
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float)
        self._loss = nn.CrossEntropyLoss(weight=weights)

    def set_classify(self):
        self.hparams.manifold = False
        self.hparams.classify = True
        self._loss = HierarchicalLoss(hparams.n_taxa_all) if self.hparams.tgt_tax_lvl == 'all' else nn.CrossEntropyLoss()

    def set_inference(self, inference=True):
        self._inference = inference

    def configure_optimizers(self):
        optimizer = None
        has_apex = True
        try:
            import apex.optimizers as aoptim
        except:
            has_apex = False

        if has_apex and self.hparams.apex:
            if self.hparams.optimizer == 'lamb':
                optimizer = aoptim.FusedLAMB(self.parameters(), lr=self.hparams.lr)
            elif self.hparams.optimizer == 'adamw':
                optimizer = aoptim.FusedAdam(self.parameters(), lr=self.hparams.lr, adam_w_mode=True)
            else:
                optimizer = aoptim.FusedAdam(self.parameters(), lr=self.hparams.lr, adam_w_mode=False)
        else:
            if self.hparams.optimizer == 'lamb':
                optimizer = ptoptim.Lamb(self.parameters(), lr=self.hparams.lr)
            elif self.hparams.optimizer == 'adamw':
                optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
            else:
                optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = None
        if self.hparams.lr_scheduler == 'cyclic':
            scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                                    base_lr=self.hparams.lr/100.0,
                                                    max_lr=self.hparams.lr,
                                                    mode='triangular2' )
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs*10)
        elif self.hparams.lr_scheduler == 'cosinewr':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                       T_0=10, T_mult=2, eta_min=1e-5)
        elif self.hparams.lr_scheduler == 'plateau':
            mode, monitor = 'min', self.val_loss
            if self.hparams.classify:
                mode, monitor = 'max', self.val_acc
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, mode=mode)
            scheduler = {
                'scheduler': scheduler,         # The LR scheduler instance (required)
                'interval': 'epoch',        # The unit of the scheduler's step size, could also be 'step'
                'frequency': 1,             # The frequency of the scheduler
                'monitor': monitor,      # Metric for `ReduceLROnPlateau` to monitor
                'strict': True,             # Whether to crash the training if `monitor` is not found
                'name': None,               # Custom name for `LearningRateMonitor` to use
            }
        elif self.hparams.lr_scheduler == 'step':
            step_size = getattr(self.hparams, 'step_size', 2)
            n_steps = getattr(self.hparams, 'n_steps', 3)
            step_factor = getattr(self.hparams, 'step_factor', 0.1)
            milestones = list(range(step_size, step_size * (n_steps + 1), step_size))
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=step_factor)

        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    @staticmethod
    def accuracy(output, target):
        pred = torch.argmax(output, dim=1)
        acc = (pred == target).float().sum()/len(target)
        return acc

    def restart(self):
        """Set training/validation epoch start time"""
        t = time()
        self.start_time = t
        self.last_time = t

    def time_stats(self, batch_idx):
        curr_time = time()
        step_time = curr_time - self.last_time
        self.last_time = curr_time
        wall_time = curr_time - TIME_OFFSET
        return {'rate': (batch_idx + 1) / (curr_time - self.start_time), 'time': step_time, 'wall_time': wall_time}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, seqs, target, olen, seq_id = batch
        return self.forward(seqs)

    # TRAIN
    def on_train_start(self):
        self.restart()

    def training_step(self, batch, batch_idx):
        seqs, target = batch
        output = self.forward(seqs)
        output = self.arc(output, target)
        loss = self._loss(output, target.long())
        if self.hparams.classify:
            self.log(self.train_acc, self.accuracy(output, target), prog_bar=True)
        stats = self.time_stats(batch_idx)
        stats[self.train_loss] = loss
        self.log_dict(stats, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs):
        return None

    # VALIDATION
    def on_validation_start(self):
        self.restart()

    def validation_step(self, batch, batch_idx):
        seqs, target = batch
        output = self.forward(seqs)
        output = self.arc(output, target)
        loss = self._loss(output, target.long())
        if self.hparams.classify:
            self.log(self.val_acc, self.accuracy(output, target), prog_bar=True)
        stats = self.time_stats(batch_idx)
        stats[self.val_loss] = loss
        self.log_dict(stats, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        return None

    # TEST
    def test_step(self, batch, batch_idx):
        seqs, target = batch
        output = self(seqs)
        loss = self._loss(output, target.long())
        self.log_dict({self.test_loss: loss, 'time': self.step_time(), 'wall_time': time()})
        return loss

    def test_epoch_end(self, outputs):
        return None
