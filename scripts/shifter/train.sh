#!/bin/bash
#SBATCH -q debug
#SBATCH -A m2865_g
#SBATCH -t 60
#SBATCH --ntasks 8
# SBATCH --ntasks 64
#SBATCH -o runs/train.%j.log
#SBATCH -e runs/train.%j.log
#SBATCH -J n2_b512_SHFTR
# SBATCH -J n16_b512_SHFTR
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4


INPUT="/pscratch/sd/a/ajtritt/exabiome/deep-taxon/input/gtdb/r207/r207.rep.h5"
SCRIPT="/global/homes/a/ajtritt/projects/exabiome/deep-taxon.git/bin/deep-taxon.py"
NODES=2

JOB="$SLURM_JOB_ID"

OUTDIR="runs/train.$JOB"
CONF="$OUTDIR.yml"

mkdir -p $OUTDIR
cp $0 $OUTDIR.sh
cp train.yml $CONF

LOG="$OUTDIR.log"

OPTIONS="--csv --slurm -g 4 -n $NODES -e 4 -k 6 -y -E shifter_n${NODES}_g4"
CMD="$SCRIPT train --sanity 100 --slurm $OPTIONS $CONF $INPUT $OUTDIR"

IMG="ajtritt/deep-taxon:amd64_v1"

shifter --image=$IMG python $CMD > $LOG 2>&1
