#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --account=m2865
#SBATCH -J train-cgpu
#SBATCH -o /global/homes/a/ajtritt/projects/exabiome/logs/%x-%j.out

EPOCHS=1000
G=4
B=32
LR=0.0001
O=256
A=4
W=4000
S=4000
SCHED=""
LOSS=M
MODEL=roznet
DATASET=medium
R=""
seed=""
CKPT=""
DEBUG=""
EXP=""

function print_help(){

    echo -e "Usage: bash train.sh [options] \n"\
            "  options:\n"\
            "    -h:   print this message\n"\
            "    -d:   print the command to be called and exit\n"\
            "    -g:   the number of GPUs to use. default $G\n"\
            "    -l:   the learning rate to use for training. default $LR\n"\
            "    -o:   the number of dimensions to output. default $O\n"\
            "    -A:   the number of batches to accumulate. default $A\n"\
            "    -W:   the size of chunks to use. default $W\n"\
            "    -S:   the chunking step size. default $S\n"\
            "    -s:   the seed to use. default is to pull system time\n"\
            "    -L:   the loss function to use. default $LOSS\n"\
            "    -M:   the model name. default $M\n"\
            "    -D:   the dataset name. default $D\n"\
            "    -e:   the number of epochs to run for. default $EPOCHS\n"\
            "    -r:   use reverse complement sequences. use only fwd strand by default\n"\
            "    -u:   the learning rate scheduler to use. default is to use train default\n"\
            "    -c:   a checkpoint file to restart from\n"\

}

while getopts "hg:b:l:O:A:W:S:L:M:D:E:e:ru:s:c:d" opt; do
  case $opt in
    h) print_help & exit 0;;
    g) G=$OPTARG ;;
    b) B=$OPTARG ;;
    l) LR=$OPTARG ;;
    o) o=$OPTARG ;;
    A) A=$OPTARG ;;
    W) W=$OPTARG ;;
    S) S=$OPTARG ;;
    u) SCHED=$OPTARG ;;
    s) seed=$OPTARG ;;
    L) LOSS=$OPTARG ;;
    M) MODEL=$OPTARG ;;
    D) DATASET=$OPTARG ;;
    e) EPOCHS=$OPTARG ;;
    r) R="-r";;
    c) CKPT=$OPTARG;;
    E) EXP=$OPTARG;;
    d) DEBUG="debug";;
  esac
done
shift $(( $OPTIND - 1))
INPUT=${1:?"Missing input file"};
INPUT=`realpath $INPUT`

OPTIONS="-$LOSS -b $B -g $G -o $O --half -W $W -S $S --lr $LR -A $A -e $EPOCHS -L"
TMP_EXP=o${O}_g${G}_b${B}_lr${LR}_16bit_A${A}

# figure out the chunking to use
CHUNKS=chunks_W${W}_S${S}
if [[ ! -z "${R}" ]]; then
    CHUNKS=${CHUNKS}
    OPTIONS="$OPTIONS -r"
else
    CHUNKS=${CHUNKS}_fwd-only
fi

# Use seed if its been passed in
if [[ ! -z "${seed}" ]]; then
    OPTIONS="$OPTIONS -s $seed"
fi

# Use a schedular if its been passed in
if [[ ! -z "${SCHED}" ]]; then
    OPTIONS="$OPTIONS --lr_scheduler $SCHED"
    TMP_EXP=${TMP_EXP}_$SCHED
fi

# Use a checkpoint if its been passed in
if [[ ! -z "${CKPT}" ]]; then
    OPTIONS="$OPTIONS -c $CKPT"
fi

# Use the experiment if its been given
if [[ ! -z "${EXP}" ]]; then
    TMP_EXP=$EXP
fi

OPTIONS="$OPTIONS -E $TMP_EXP"

OUTDIR=${2:-$CSCRATCH/exabiome/deep-index/train/datasets/$DATASET/$CHUNKS/$MODEL/$LOSS}
LOG=$OUTDIR/logs/$TMP_EXP.log

CMD="deep-index train $OPTIONS $MODEL $INPUT $OUTDIR"

if [[ ! -z "${DEBUG}" ]]; then
    echo $CMD
    echo $LOG
else
    echo $OPTIONS
    echo $MODEL
    echo $INPUT
    echo $OUTDIR
    mkdir -p $OUTDIR/logs
    # Setup software
    module load python
    conda activate exabiome_16bit
    # Run the training
    srun -u $CMD > $LOG 2>&1
fi
