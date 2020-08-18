#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --account=m2865
#SBATCH -J summarize
#SBATCH -o /global/homes/a/ajtritt/projects/exabiome/logs/%x-%j.out

# Setup software
module load python
conda activate exabiome_16bit

DI_DIR=$CSCRATCH/exabiome/deep-index
DATASET=medium

MODEL=${1:?"Missing model name"}
CKPT=${2:?"Missing checkpoint file"}
INPUT=${3:?"Missing input dataset"}

OUTDIR=`dirname $CKPT`/`basename $CKPT .ckpt`
mkdir -p $OUTDIR

echo "Running inference on $CKPT"

# Run inference
INF_LOG=$OUTDIR/infer.log
srun -u deep-index infer $MODEL $INPUT $CKPT -g 1 -b 32 -L -U >> $INF_LOG 2>&1

# Summarize results
OUTPUTS=$OUTDIR/outputs.h5
SUM_LOG=$OUTDIR/summarize.log
srun -u deep-index summarize $OUTPUTS > $SUM_LOG 2>&1
