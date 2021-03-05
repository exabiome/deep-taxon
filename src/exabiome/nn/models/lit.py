from .. import train_test_loaders
from pytorch_lightning import LightningModule
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
    test_loss = 'test_loss'

    schedules = ('adam', 'cyclic', 'plateau')

    def __init__(self, hparams):
        super().__init__()
        self.hparams = self.check_hparams(hparams)
        if self.hparams.manifold:
            self._loss = DistMSELoss()
        elif self.hparams.classify:
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

    def _check_loaders(self):
        """
        Load dataset if it has not been loaded yet
        """
        if not hasattr(self, 'loaders'):
            dataset, io = process_dataset(self.hparams, inference=self._inference)
            if self.hparams.load:
                dataset.load()

            kwargs = dict(random_state=self.hparams.seed,
                          batch_size=self.hparams.batch_size,
                          distances=self.hparams.manifold)
            kwargs.update(self.hparams.loader_kwargs)
            if self._inference:
                kwargs['distances'] = False
                kwargs.pop('num_workers', None)
                kwargs.pop('multiprocessing_context', None)
            tr, te, va = train_test_loaders(dataset, **kwargs)
            self.loaders = {'train': tr, 'test': te, 'validate': va}
            self.dataset = dataset

    def configure_optimizers(self):
        if self.hparams.lr_scheduler == 'adam':
            return optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
            scheduler = None
            if self.hparams.lr_scheduler == 'cyclic':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max=self.hparams.lr)
            elif self.hparams.lr_scheduler == 'plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return [optimizer], [scheduler]

    # TRAIN
    def train_dataloader(self):
        self._check_loaders()
        return self.loaders['train']

    def training_step(self, batch, batch_idx):
        idx, seqs, target, olen, seq_id = batch
        output = self.forward(seqs)
        loss = self._loss(output, target)
        self.log(self.train_loss, loss)
        return loss

    def training_epoch_end(self, outputs):
        return None

    # VALIDATION
    def val_dataloader(self):
        self._check_loaders()
        return self.loaders['validate']

    def validation_step(self, batch, batch_idx):
        idx, seqs, target, olen, seq_id = batch
        output = self(seqs)
        loss = self._loss(output, target)
        self.log(self.val_loss, loss)
        return loss

    def validation_epoch_end(self, outputs):
        return None

    # TEST
    def test_dataloader(self):
        self._check_loaders()
        return self.loaders['test']

    def test_step(self, batch, batch_idx):
        idx, seqs, target, olen, seq_id = batch
        output = self(seqs)
        loss = self._loss(output, target)
        self.log(self.test_loss, loss)
        return loss

    def test_epoch_end(self, outputs):
        return None
