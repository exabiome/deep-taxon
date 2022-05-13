#!/bin/bash
#SBATCH -A m2865_g
#SBATCH -t 360
#SBATCH --ntasks 64
#SBATCH -o train.%j.log
#SBATCH -e train.%j.log
#SBATCH -J n16_b512_SHFTR
#SBATCH -C gpu
#SBATCH -c 16
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH --image=ajtritt/deep-taxon:amd64_v1


INPUT="$PSCRATCH/exabiome/deep-taxon/input/gtdb/r207/r207.rep.h5"
REPO_DIR="$HOME/projects/exabiome/deep-taxon.git"

SCRIPT="$REPO_DIR/bin/deep-taxon.py"
NODES=16

JOB="$SLURM_JOB_ID"

OUTDIR="runs/train.$JOB"
CONF="$OUTDIR.yml"

mkdir -p $OUTDIR
cp $0 $OUTDIR.sh
cp $REPO_DIR/configs/graphcore.yml $CONF

LOG="$OUTDIR.log"

OPTIONS="--csv --slurm -g 4 -n $NODES -e 6 -k 6 -y -D -E shifter_n${NODES}_g4"
CMD="$SCRIPT train $OPTIONS $CONF $INPUT $OUTDIR"

mv train.$JOB.log $LOG
srun --ntasks $(($NODES*4)) shifter python $CMD > $LOG 2>&1
