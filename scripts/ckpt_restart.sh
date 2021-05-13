#!/bin/bash
#SBATCH -q regular
#SBATCH -A m2865
#SBATCH -t 3:59:59
#SBATCH -n 8
#SBATCH -o ./train/datasets/full/chunks_W4000_S4000/resnet_clf/C/n1_g1_b16_r0.001/train.%j.log
#SBATCH -e ./train/datasets/full/chunks_W4000_S4000/resnet_clf/C/n1_g1_b16_r0.001/train.%j.log
#SBATCH -J n1_g1_b16_r0.001
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1



# FEATS_CKPT="train/datasets/full/chunks_W4000_S4000/resnet18_feat/M/n1_g1_b16_r0.001/train.1570147/epoch=10-step=127368.ckpt"


INPUT=${1:?"Please provide an input file"}
FEATS_CKPT=${2:?"Please provide a checkpoint file for features"}
CONDA_ENV=${3:?"Please provide the conda environment ot use"}

conda activate $CONDA_ENV 

MAIN_OUTDIR="./test_ckpt_restart/datasets/full/chunks_W4000_S4000/resnet_clf/C/n1_g1_b16_r0.001"
mkdir -p $MAIN_OUTDIR

JOB="TESTCKPT.0"
OPTIONS=" --profile -C -b 16 -d -g 1 -n 1 -o 256 -W 4000 -S 4000 -r 0.001 -A 1 -e 30 -l -s 2222 -F $FEATS_CKPT -E n1_g1_b16_r0.001"
OUTDIR="$MAIN_OUTDIR/train.$JOB"
LOG="$OUTDIR.log"
CMD="deep-index train --slurm $OPTIONS resnet_clf $INPUT $OUTDIR"

cp $0 $OUTDIR.sh
mkdir -p $OUTDIR
echo "$CMD > $LOG"
srun $CMD > $LOG 2>&1

JOB="TESTCKPT.1"
CKPT=`ls $OUTDIR/epoch*ckpt`
OPTIONS=" --profile -C -b 16 -g 1 -d -n 1 -o 256 -W 4000 -S 4000 -r 0.001 -A 1 -e 30 -l -s 2222 -c $CKPT -E n1_g1_b16_r0.001"
OUTDIR="$MAIN_OUTDIR/train.$JOB"
LOG="$OUTDIR.log"
CMD="deep-index train --slurm $OPTIONS resnet_clf $INPUT $OUTDIR"

cp $0 $OUTDIR.sh
mkdir -p $OUTDIR
echo "$CMD > $LOG"
srun $CMD > $LOG 2>&1

JOB="TESTCKPT.2"
CKPT=`ls $OUTDIR/epoch*ckpt`
OPTIONS=" --profile -C -b 16 -g 1 -d -n 1 -o 256 -W 4000 -S 4000 -r 0.001 -A 1 -e 30 -l -s 2222 -c $CKPT -E n1_g1_b16_r0.001"
OUTDIR="$MAIN_OUTDIR/train.$JOB"
LOG="$OUTDIR.log"
CMD="deep-index train --slurm $OPTIONS resnet_clf $INPUT $OUTDIR"

cp $0 $OUTDIR.sh
mkdir -p $OUTDIR
echo "$CMD > $LOG"
srun $CMD > $LOG 2>&1

JOB="TESTCKPT.3"
CKPT=`ls $OUTDIR/epoch*ckpt`
OPTIONS=" --profile -C -b 16 -g 1 -d -n 1 -o 256 -W 4000 -S 4000 -r 0.001 -A 1 -e 30 -l -s 2222 -c $CKPT -E n1_g1_b16_r0.001"
OUTDIR="$MAIN_OUTDIR/train.$JOB"
LOG="$OUTDIR.log"
CMD="deep-index train --slurm $OPTIONS resnet_clf $INPUT $OUTDIR"

cp $0 $OUTDIR.sh
mkdir -p $OUTDIR
echo "$CMD > $LOG"
srun $CMD > $LOG 2>&1
