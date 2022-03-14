#!/bin/bash
#SBATCH --qos=regular
#SBATCH -A m3513
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=10
#SBATCH -o ./array_logs/prodigal_array.%j.log
#SBATCH -e ./array_logs/prodigal_array.%j.log
#SBATCH --array=1-8059:32
#SBATCH -J prodigal_array

module load parallel
conda activate cat

STEP=32
WD=$PWD
DIR="./array_logs/prodigal_array.${SLURM_ARRAY_JOB_ID}/job.${SLURM_JOB_ID}_{$SLURM_ARRAY_TASK_ID}"
mkdir -p $DIR
cd $DIR
let "END = $STEP + $SLURM_ARRAY_TASK_ID - 1"
sed -n "${SLURM_ARRAY_TASK_ID},${END}p" $WD/untranslated.txt | parallel --jobs $STEP bash $WD/prodigal.sh
cd $WD
