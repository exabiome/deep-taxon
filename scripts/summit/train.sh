#!/bin/bash
# Begin LSF Directives
#BSUB -q debug
#BSUB -P BIF115
#BSUB -W 1:00
#BSUB -nnodes 2
#BSUB -alloc_flags "gpumps NVME"
#BSUB -J DeepIndexTrain
#BSUB -o DeepIndexTrain.%J
#BSUB -e DeepIndexTrain.%J

EPOCHS=1000
G=6
N=2
B=32
LR=0.0001
O=256
A=1
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
            "    -n:   the number of nodes to use. default $N\n"\
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

while getopts "hn:g:b:l:O:A:W:S:L:M:D:E:e:ru:s:c:d" opt; do
  case $opt in
    h) print_help & sleep 1s & exit 0;;
    g) G=$OPTARG ;;
    n) N=$OPTARG ;;
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

OPTIONS=""
if [[ ! -z "${DEBUG}" ]]; then
    OPTIONS="$OPTIONS -d"
fi

#OPTIONS="$OPTIONS -$LOSS -b $B -g $G -n $N -o $O --half -W $W -S $S --lr $LR -A $A -e $EPOCHS -L"
OPTIONS="$OPTIONS -$LOSS -b $B -g $G -n $N -o $O -W $W -S $S --lr $LR -A $A -e $EPOCHS -L"
TMP_EXP=n${N}_g${G}_A${A}_b${B}_lr${LR}_o${O}

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

JOB=${LSB_JOBID:-NOSUB}
OUTDIR=${2:-/gpfs/alpine/bif115/scratch/ajtritt/deep-index/train/datasets/$DATASET/$CHUNKS/$MODEL/$LOSS/$TMP_EXP/$JOB}
LOG=$OUTDIR/log

BB_INPUT=/mnt/bb/$USER/`basename $INPUT`

CMD="deep-index train --summit $OPTIONS $MODEL $BB_INPUT $OUTDIR"

echo options=$OPTIONS
echo model=$MODEL
echo input=$INPUT
echo outdir=$OUTDIR

if [[ $JOB == "NOSUB" ]]; then
    echo "$CMD > $LOG"
else
    echo "$INPUT to $BB_INPUT"
    cp $INPUT $BB_INPUT
    mkdir -p $OUTDIR
    # Setup software
    module load ibm-wml-ce/1.7.1.a0-0
    conda activate exabiome-wml
    # Run the training
    #jsrun -n $N -r 6 -a 7 -K 3 -g 1 $CMD > $LOG 2>&1
    ddlrun $CMD > $LOG 2>&1
fi
