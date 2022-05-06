import os
import sys

import numpy as np
import pytorch_lightning.cluster_environments as cenv
import torch
from torch import distributed as dist



def test_dist(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('system', options=['lsf', 'slurm'], help='which system the check is being run on')
    args = parser.parse_args()

    env = None
    if args.lsf:
        env = cenv.LSFEnvironment()
    else:
        env = cenv.SLURMEnvironment()

    backend = "nccl"
    os.environ['MASTER_ADDR'] = env.master_addr() # os.environ['LSB_HOSTS'].split()[1]
    os.environ['MASTER_PORT'] = env.master_port() # str(int(os.environ['LSB_JOBID']) + 10000)
    rank = env.global_rank() # int(os.environ['JSM_NAMESPACE_RANK'])
    world_size = env.world_size() # int(os.environ['JSM_NAMESPACE_SIZE'])
    rank_id = "RANK-%s/%s" % (rank, world_size)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(rank_id, "Finished initializing process group; backend: %s, rank: %d, "
    "world_size: %d" % (backend, rank, world_size), file=sys.stderr)


    device_id = env.local_rank() # int(os.environ['JSM_NAMESPACE_LOCAL_RANK'])

    a = torch.from_numpy(np.random.rand(3, 3)).cuda(device_id)
    dist.broadcast(tensor=a, src=0)
    print(rank_id, 'Successfully broadcastted', file=sys.stderr)
