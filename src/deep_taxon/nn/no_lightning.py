from collections import deque
import os
import shutil
from time import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tqdm import tqdm

class RunningAverage:

    def __init__(self, max_samples=None):
        self._avg = 0.0
        self._n = 0
        self.max_samples = max_samples
        self._samples = deque()

    @property
    def avg(self):
        if self.max_samples:
            return sum(self._samples)/len(self._samples)
        else:
            return self._avg

    def update(self, loss):
        loss = float(loss)
        if self.max_samples:
            self._samples.append(loss)
            if len(self._samples) > self.max_samples:
                self._samples.popleft()
        else:
            self._avg =  (self._n * self._avg + loss) / (self._n + 1)
            self._n += 1

    def __repr__(self):
        return str(self.avg)

    def __format__(self, s):
        return format(self.avg, s)

    def __float__(self):
        return self.avg

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
        self.metrics_path = f"{output_dir}/metrics.csv"

        self.best_path = None
        self.best_loss = float('inf')
        self.step = 0
        self.output_dir = output_dir

        if os.path.exists(self.snapshot_path):
            print(f"Loading snapshot from {self.snapshot_path}")
            self._load_snapshot(self.snapshot_path)

        self._n_loss_samples = 100
        self._log_interval = self._n_loss_samples

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.best_loss = snapshot["LOSS"]
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer = snapshot["OPTIMIZER"]
        self.step = snapshot['STEP']
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

    def _run_loop(self, epoch, it):
        avg = RunningAverage(self._n_loss_samples)
        time_avg = RunningAverage(self._n_loss_samples)
        s = time()
        for i, (source, targets) in enumerate(it):
            self.step += 1
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            avg.update(self._run_batch(source, targets))
            if isinstance(it, tqdm):
                it.set_postfix(loss=f"{avg:0.8f}")
            e = time()
            time_avg.update(e - s)
            s = e
            if self.global_rank == 0 and i % self._log_interval == 0:
                phase = 'train' if torch.is_grad_enabled() else 'val'
                print(",".join([str(x) for x in (self.step, phase, avg, time_avg)]), file=self.metrics_file)
        return float(avg)


    @torch.enable_grad()
    def _run_train(self, epoch):
        if self.train_data is None:
            self.train_data = self.data_mod.train_dataloader()
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if self.global_rank == 0:
            it = tqdm(self.train_data, desc=f"Epoch {epoch}") #, leave=False)
        avg_loss = self._run_loop(epoch, it)
        if self.scheduler is not None:
            self.scheduler.step()
        return avg_loss

    @torch.no_grad()
    def _run_validate(self, epoch):
        if self.validate_data is None:
            self.validate_data = self.data_mod.val_dataloader()
        if self.global_rank == 0:
            it = tqdm(self.validate_data, desc=f"Validation epoch {epoch}") #, leave=False)
        avg_loss = self._run_loop(epoch, it)
        return avg_loss

    def _save_snapshot(self, epoch, valid_loss, step):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch + 1,
            "OPTIMIZER": self.optimizer,
            "LOSS": valid_loss,
            "STEP": step,
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
        if self.global_rank == 0:
            if os.path.exists(self.metrics_path):
                self.metrics_file = open(self.metrics_path, 'a')
            else:
                self.metrics_file = open(self.metrics_path, 'w')
                print("step,phase,loss,rate", file=self.metrics_file)

        for epoch in range(self.epochs_run, max_epochs):
            self._run_train(epoch)
            valid_loss = self._run_validate(epoch)
            if self.global_rank == 0:
                self._save_snapshot(epoch, valid_loss, self.step)
            dist.barrier()
        self.model = self.model.module
        if self.global_rank == 0:
            self.metrics_file.close()
        dist.destroy_process_group()
