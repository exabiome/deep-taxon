import os
import socket
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin


class LSFEnvironment(ClusterEnvironment):
    """An environment for running on clusters managed by the LSF resource manager.

    It is expected that any execution using this ClusterEnvironment was executed
    using the Job Step Manager i.e. jsrun.

    This plugin expects the following environment variables:

    LSB_JOBID
      The LSF assigned job ID

    LSB_HOSTS
      The hosts used in the job. This string is expected to have the format "batch <rank_0_host> ...."

    JSM_NAMESPACE_LOCAL_RANK
      The node local rank for the task. This environment variable is set by jsrun

    JSM_NAMESPACE_SIZE
      The world size for the task. This environment variable is set by jsrun
    """

    def __init__(self):
        self._master_address = self._get_master_address()
        self._master_port = self._get_master_port()
        self._local_rank = self._get_local_rank()
        self._global_rank = self._get_global_rank()
        self._world_size = self._get_world_size()
        self._node_rank = self._get_node_rank()

        # set environment variables needed for initializing torch distributed process group
        os.environ["MASTER_ADDR"] = str(self._master_address)
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        os.environ["MASTER_PORT"] = str(self._master_port)
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        self._rep = ",".join('%s=%s' % (s, getattr(self, "_"+s)) for s in ('master_address', 'master_port', 'world_size', 'local_rank', 'node_rank', 'global_rank'))

    def _read_hosts(self):
        var = "LSB_HOSTS"
        hosts = os.environ.get(var)
        if not hosts:
            raise ValueError("Could not find hosts -- expected in environment variable %s" % var)
        hosts = hosts.split()
        if len(hosts) < 2:
            raise ValueError("Cannot parse hosts from LSB_HOSTS environment variable -- "
                             "expected format \"batch <rank_0_host> ...\"")
        return hosts

    def _get_master_address(self):
        """A helper for getting the master address"""
        hosts = self._read_hosts()
        return hosts[1]

    def _get_master_port(self):
        """A helper for getting the master port

        Use the LSF job ID so all ranks can compute the master port
        """
        # check for user-specified master port
        port = os.environ.get("MASTER_PORT")
        if not port:
            var = "LSB_JOBID"
            jobid = os.environ.get(var)
            if not jobid:
                raise ValueError("Could not find job id -- expected in environment variable %s" % var)
            else:
                port = int(jobid)
                # all ports should be in the 10k+ range
                port = int(port) % 1000 + 10000
            log.debug("calculated master port")
        else:
            log.debug("using externally specified master port")
        return port

    def _get_global_rank(self):
        """A helper function for getting the global rank

        Read this from the environment variable JSM_NAMESPACE_LOCAL_RANK
        """
        var = "JSM_NAMESPACE_RANK"
        global_rank = os.environ.get(var)
        if global_rank is None:
            raise ValueError("Cannot determine global rank -- expected in %s "
                             "-- make sure you run your executable with jsrun" % var)
        return int(global_rank)

    def _get_local_rank(self):
        """A helper function for getting the local rank

        Read this from the environment variable JSM_NAMESPACE_LOCAL_RANK
        """
        var = "JSM_NAMESPACE_LOCAL_RANK"
        local_rank = os.environ.get(var)
        if local_rank is None:
            raise ValueError("Cannot determine local rank -- expected in %s "
                             "-- make sure you run your executable with jsrun" % var)
        return int(local_rank)

    def _get_world_size(self):
        """A helper function for getting the world size

        Read this from the environment variable JSM_NAMESPACE_SIZE
        """
        var = "JSM_NAMESPACE_SIZE"
        world_size = os.environ.get(var)
        if world_size is None:
            raise ValueError("Cannot determine local rank -- expected in %s "
                             "-- make sure you run your executable with jsrun" % var)
        return int(world_size)

    def _get_node_rank(self):
        """A helper function for getting the node rank"""
        hosts = self._read_hosts()
        count = dict()
        for host in hosts:
            if 'batch' in host or 'login' in host:
                continue
            if host not in count:
                count[host] = len(count)
        return count[socket.gethostname()]


    def __str__(self):
        return self._rep

    def creates_children(self):
        """
        LSF creates subprocesses -- i.e. PyTorch Lightning does not need to spawn them
        """
        return True

    def master_address(self):
        """
        Master address is read from a list of hosts contained in the environment variable *LSB_HOSTS*
        """
        return self._master_address

    def master_port(self):
        """
        Master port is calculated from the LSF job ID
        """
        return self._master_port

    def world_size(self):
        """
        World size is read from the environment variable JSM_NAMESPACE_SIZE
        """
        return self._world_size

    def local_rank(self):
        """
        World size is read from the environment variable JSM_NAMESPACE_LOCAL_RANK
        """
        return self._local_rank

    def node_rank(self):
        """
        Node rank is determined by the position of the current hostname in the list of hosts stored in LSB_HOSTS
        """
        return self._node_rank

    def global_rank(self):
        """
        World size is read from the environment variable JSM_NAMESPACE_RANK
        """
        return self._global_rank

    def set_world_size(self, size: int) -> None:
        log.debug("SLURMEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def set_global_rank(self, rank: int) -> None:
        log.debug("SLURMEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")



class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST(os.getcwd(), download=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)

# init model
autoencoder = LitAutoEncoder()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)

parallel_devices = [torch.device(i) for i in range(torch.cuda.device_count())]
acc = GPUAccelerator(
        precision_plugin = NativeMixedPrecisionPlugin(),
        training_type_plugin = DDPPlugin(parallel_devices=parallel_devices,
                                         cluster_environment=LSFEnvironment()))

targs = {'max_epochs': 1, 'num_nodes': 2, 'accumulate_grad_batches': 1, 'gpus': 6,
         'accelerator': acc,
         'limit_train_batches': 10, 'limit_val_batches': 5, 'log_every_n_steps': 1}


# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer(**targs)
trainer.fit(autoencoder, train_loader)
