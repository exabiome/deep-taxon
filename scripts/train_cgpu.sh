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

DI_DIR=$CSCRATCH/exabiome/deep-index/
LOSS=M
MODEL=roznet
EXP=g4_b32_lr0.001
outdir=datasets/medium/chunks/$MODEL/M/training_results/

# Run the training
srun -u deep-index train \
            $MODEL $DI_DIR/input/ar122_r89.genomic.small.deep_index.input.h5 $DI_DIR/train/datasets/medium/chunks/$MODEL/$LOSS \
            -$LOSS -g 4 -b 32 -s 3001 -S 1000 -W 1000 --lr 0.001 -e 10 -E $EXP \
            > $DI_DIR/train/datasets/medium/chunks/$MODEL/$LOSS/logs/$EXP.log 2>&1
