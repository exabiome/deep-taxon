from .. import train_test_loaders
from pytorch_lightning import LightningModule
import torch_optimizer as ptoptim
import torch.optim as optim
import torch.nn as nn
import torch
import argparse

#from .. import SeqDataset
#from hdmf.common import get_hdf5io
from ..loader import process_dataset

from ...sequence import WindowChunkedDIFile
from ..loss import DistMSELoss

class AbstractLit(LightningModule):

    val_loss = 'validation_loss'
    train_loss = 'training_loss'
    val_acc = 'validation_acc'
    train_acc = 'training_acc'
    test_loss = 'test_loss'

    schedules = ('adam', 'cyclic', 'plateau')

    def __init__(self, hparams):
        super().__init__()
        #self.hparams = self.check_hparams(hparams)
        self.save_hyperparameters(hparams)
        if self.hparams.manifold:
            self._loss = DistMSELoss()
        elif self.hparams.classify:
            if self.hparams.tgt_tax_lvl == 'all':
                self._loss = HierarchicalLoss(hparams.n_taxa_all)
            else:
                self._loss = nn.CrossEntropyLoss()
        else:
            self._loss =  nn.MSELoss()
        self.set_inference(False)
        self.lr = getattr(hparams, 'lr', None)

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

    def set_dataset(self, dataset, load=True, inference=False):
        kwargs = dict(random_state=self.hparams.seed,
                      batch_size=self.hparams.batch_size,
                      distances=self.hparams.manifold)

        if inference:
            kwargs['distances'] = False
        tr, te, va = train_test_loaders(dataset, **kwargs)
        self.loaders = {'train': tr, 'test': te, 'validate': va}

    def configure_optimizers(self):
        optimizer = None
        if self.hparams.optimizer == 'lamb':
            optimizer = ptoptim.Lamb(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
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
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            scheduler = {
                'scheduler': scheduler,         # The LR scheduler instance (required)
                'interval': 'epoch',        # The unit of the scheduler's step size, could also be 'step'
                'frequency': 1,             # The frequency of the scheduler
                'monitor': self.val_loss,      # Metric for `ReduceLROnPlateau` to monitor
                'strict': True,             # Whether to crash the training if `monitor` is not found
                'name': None,               # Custom name for `LearningRateMonitor` to use
            }

        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    @staticmethod
    def accuracy(output, target):
        pred = torch.argmax(output, dim=1)
        acc = (pred == target).float().sum()/len(target)
        return acc

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, seqs, target, olen, seq_id = batch
        return self.forward(seqs)

    # TRAIN
    def training_step(self, batch, batch_idx):
        idx, seqs, target, olen, seq_id = batch
        output = self.forward(seqs)
        loss = self._loss(output, target)
        if self.hparams.classify:
            self.log(self.train_acc, self.accuracy(output, target))
        self.log(self.train_loss, loss)
        return loss

    def training_epoch_end(self, outputs):
        return None

    # VALIDATION
    def validation_step(self, batch, batch_idx):
        idx, seqs, target, olen, seq_id = batch
        output = self(seqs)
        loss = self._loss(output, target)
        if self.hparams.classify:
            self.log(self.val_acc, self.accuracy(output, target))
        self.log(self.val_loss, loss)
        return loss

    def validation_epoch_end(self, outputs):
        return None

    # TEST
    def test_step(self, batch, batch_idx):
        idx, seqs, target, olen, seq_id = batch
        output = self(seqs)
        loss = self._loss(output, target)
        self.log(self.test_loss, loss)
        return loss

    def test_epoch_end(self, outputs):
        return None
