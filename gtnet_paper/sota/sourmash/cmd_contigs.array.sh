#!/bin/bash
#SBATCH -J sourmash_ctgs
#SBATCH -A m3513
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 10
#SBATCH -o /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/contigs/sourmash_log/%A_%a.log
#SBATCH -e /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/contigs/sourmash_log/%A_%a.log
#SBATCH -n 1
#SBATCH --array=2-40

# --array=2-40
#
module load parallel

seq -f "%04g" $SLURM_ARRAY_TASK_ID 40 2000  | parallel -j 50 bash cmd_contigs.sh {}
