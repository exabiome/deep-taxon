import os
import shutil

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tqdm import tqdm

class Trainer:
    def __init__(self, model, targs, data_mod, output_dir):

        self.local_rank = targs['local_rank']
        self.global_rank = targs['global_rank']
        self.world_size = targs['world_size']
        self.model = model.to(self.local_rank)
        self.loss = model._loss
        self.data_mod = data_mod
        self.train_data = None     # This is a datalaoder
        self.validate_data = None  # This is a datalaoder
        self.optimizer = model.configure_optimizers()
        self.scheduler = None
        if isinstance(self.optimizer, tuple):
            self.scheduler = self.optimizer[1][0]
            self.optimizer = self.optimizer[0][0]
        self.epochs_run = 0
        self.snapshot_path = f"{output_dir}/last.ckpt"

        self.best_path = None
        self.best_loss = float('inf')
        self.output_dir = output_dir

        if os.path.exists(self.snapshot_path):
            print(f"Loading snapshot from {self.snapshot_path}")
            self._load_snapshot(snapshot_path)


    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.best_loss = snapshot["LOSS"]
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer = snapshot["OPTIMIZER"]
        self.scheduler = snapshot.get("SCHEDULER", None)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss(output, targets.long())
        if torch.is_grad_enabled():
            loss.backward()
            self.optimizer.step()
        return loss

    @torch.enable_grad()
    def _run_train(self, epoch):
        if self.train_data is None:
            self.train_data = self.data_mod.train_dataloader()
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        it = self.train_data
        if self.global_rank == 0:
            it = tqdm(it, desc=f"Epoch {epoch}", leave=False)
        for source, targets in it:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)
        if self.scheduler is not None:
            self.scheduler.step()

        return None

    @torch.no_grad()
    def _run_validate(self, epoch):
        if self.validate_data is None:
            self.validate_data = self.data_mod.validate_dataloader()
        it = self.validate_data
        if self.global_rank == 0:
            it = tqdm(it, desc=f"Validation epoch {epoch}", leave=False)
        for source, targets in it:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)
        return None

    def _save_snapshot(self, epoch, valid_loss):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER": self.optimizer,
            "LOSS": valid_loss,
        }
        if self.scheduler is not None:
            snapshot['SCHEDULER'] = self.scheduler
        torch.save(snapshot, self.snapshot_path)

        # save best checkpoint
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            to_rm = self.best_path
            self.best_path = f"{self.output_dir}/best_epoch{epoch:03}.ckpt"
            shutil.copyfile(self.snapshot_path, self.best_path)
            if to_rm is not None:
                os.remove(to_rm)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        dist.init_process_group(backend="nccl", rank=self.global_rank, world_size=self.world_size)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        for epoch in range(self.epochs_run, max_epochs):
            self._run_train(epoch)
            valid_loss = self._run_validate()
            if self.local_rank == 0:
                self._save_snapshot(epoch, valid_loss)
            dist.barrier()
        self.model = self.model.model
        dist.destroy_process_group()
