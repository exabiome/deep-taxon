#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --account=m2865
#SBATCH -J train-cgpu
#SBATCH -o /global/homes/a/ajtritt/projects/exabiome/logs/%x-%j.out

E=1000
G=4
B=32
LR=0.0001
O=256
A=4
W=4000
S=4000
LOSS=M
MODEL=roznet
DATASET=medium

function print_help(){

    echo -e "Usage: bash train.sh [options] \n"\
            "  options:\n"\
            "    -h:   print this message\n"\
            "    -g:   the number of GPUs to use. default $G\n"\
            "    -l:   the learning rate to use for training. default $LR\n"\
            "    -o:   the number of dimensions to output. default $O\n"\
            "    -A:   the number of batches to accumulate. default $A\n"\
            "    -W:   the size of chunks to use. default $W\n"\
            "    -S:   the chunking step size. default $S\n"\
            "    -L:   the loss function to use. default $L\n"\
            "    -M:   the model name. default $M\n"\
            "    -D:   the dataset name. default $D\n"\
            "    -E:   the number of epochs to run for. default $E\n"\

}

while getopts "hg:b:l:O:A:W:S:L:M:D:E:" opt; do
  case $opt in
    h) print_help & exit 0;;
    g) G=$OPTARG ;;
    b) B=$OPTARG ;;
    l) LR=$OPTARG ;;
    o) o=$OPTARG ;;
    A) A=$OPTARG ;;
    W) W=$OPTARG ;;
    S) S=$OPTARG ;;
    L) LOSS=$OPTARG ;;
    M) MODEL=$OPTARG ;;
    D) DATASET=$OPTARG ;;
    E) E=$OPTARG ;;
  esac
done
shift $(( $OPTIND - 1))
INPUT=${1:?"Missing input file"};
INPUT=`realpath $INPUT`

EXP=o${O}_g${G}_b${B}_lr${LR}_16bit_A${A}
OUTDIR=${2:-$CSCRATCH/exabiome/deep-index/train/datasets/$DATASET/chunks_W${W}_S${S}/$MODEL/$LOSS}

mkdir -p $OUTDIR/logs

# Setup software
module load python
conda activate exabiome_16bit

# Run the training
srun -u deep-index train -M -b $B -g $G -o $O -s $S --half -W $W -S $S --lr $LR -A $A -E $EXP -e $E -L \
                         $MODEL $INPUT $OUTDIR > $OUTDIR/logs/$EXP.log 2>&1
