from .. import train_test_loaders
from pytorch_lightning import LightningModule
import torch.optim as optim
import torch.nn as nn
import torch

class AbstractLit(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def set_dataset(self, dataset):
        tr, te, va = train_test_loaders(dataset,
                                        random_state=self.hparams.seed,
                                        batch_size=self.hparams.batch_size,
                                        downsample=self.hparams.downsample)
        self.loaders = {'train': tr, 'test': te, 'validate': va}

    def set_loss(self, criterion):
        self._loss = criterion

    def train_dataloader(self):
        return self.loaders['train']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        #schedular = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        #return optimizer, schedular
        return optimizer

    def training_step(self, batch, batch_idx):
        idx, seqs, target, olen = batch
        output = self.forward(seqs)
        loss = self._loss(output, target)
        return {'loss': loss}

    def val_dataloader(self):
        return self.loaders['validate']

    def validation_step(self, batch, batch_idx):
        idx, seqs, target, olen = batch
        output = self(seqs)
        return {'val_loss': self._loss(output, target)}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def test_dataloader(self):
        return self.loaders['test']

    def test_step(self, batch, batch_idx):
        idx, seqs, target, olen = batch
        output = self(seqs)
        return {'test_loss': self._loss(output, target)}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}


