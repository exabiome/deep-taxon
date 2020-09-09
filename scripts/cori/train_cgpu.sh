#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
# SBATCH --account=m2865
# SBATCH --ntasks-per-node=8
# SBATCH --gpus-per-task=1
# SBATCH --exclusive
#SBATCH -J train-cgpu
#SBATCH -o /global/homes/a/ajtritt/projects/exabiome/logs/%x-%j.out

# Setup software
module load python
conda activate exabiome_38

G=4
B=32
LR=0.001
EXP=g${G}_b${B}_lr${LR}

DI_DIR=$CSCRATCH/exabiome/deep-index/
LOSS=M
MODEL=roznet
DATASET=medium
OUTDIR=$DI_DIR/train/datasets/$DATASET/chunks/$MODEL/$LOSS
mkdir -p $OUTDIR/logs

# Run the training
srun -u deep-index train \
            $MODEL $DI_DIR/input/ar122_r89.genomic.$DATASET.deep_index.input.h5 $OUTDIR \
            -$LOSS -g $G -b $B -s 3001 -S 1000 -W 1000 --lr $LR -e 50 -E $EXP \
            > $OUTDIR/logs/$EXP.log 2>&1
