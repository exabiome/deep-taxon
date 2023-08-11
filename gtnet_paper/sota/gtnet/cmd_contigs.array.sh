#!/bin/bash
#SBATCH -J gtnet_contigs
#SBATCH -A m2865
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 60
#SBATCH -o /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/contigs/gtnet_log/%A_%a.log
#SBATCH -e /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/contigs/gtnet_log/%A_%a.log
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 32
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH --array=1-499


srun bash cmd_contigs.sh `seq -f "%04g" $SLURM_ARRAY_TASK_ID 500 1999`
