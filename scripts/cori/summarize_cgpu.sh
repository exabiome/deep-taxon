#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
# SBATCH --account=m2865
# SBATCH --ntasks-per-node=8
# SBATCH --gpus-per-task=1
# SBATCH --exclusive
#SBATCH -J train-cgpu
#SBATCH -o /global/homes/a/ajtritt/projects/exabiome/logs/%x-%j.out

# Setup software
module load python
conda activate exabiome_16bit

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


# Run inference
srun -u deep-index infer \
            $MODEL $DI_DIR/input/ar122_r89.genomic.$DATASET.deep_index.input.h5 $OUTDIR \
            -g 1 -b 256 -E $EXP -L -U \
            # -c $OUTDIR/training_results/$EXP/seed=3001-epoch=44-val_loss=3.84.ckpt   # You might need to add this argument if there are more than one checkpoint files
            > $OUTDIR/logs/$EXP.infer.log 2>&

# Summarize results
srun -u deep-index summarize $OUTDIR/training_results/$EXP > $OUTDIR/logs/$EXP.summarize.log 2>&
