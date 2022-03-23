#!/bin/bash
#SBATCH --qos=regular
#SBATCH -A m3513
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=10
#SBATCH -o ./array_logs/find_orfs_array.%j.log
#SBATCH -e ./array_logs/find_orfs_array.%j.log
#SBATCH --array=1-64:32
#SBATCH -J find_orfs_array

# originally --array=1-26084:32

module load parallel
conda activate cat

STEP=32
WD=$PWD
DIR="./array_logs/find_orfs_array.${SLURM_ARRAY_JOB_ID}/job.${SLURM_JOB_ID}_{$SLURM_ARRAY_TASK_ID}"
mkdir -p $DIR
cd $DIR
let "END = $STEP + $SLURM_ARRAY_TASK_ID - 1"
sed -n "${SLURM_ARRAY_TASK_ID},${END}p" $WD/leftovers.txt | parallel --jobs $STEP bash $WD/find_orfs.sh
# sed -n "${SLURM_ARRAY_TASK_ID},${END}p" $WD/r202.nonrep.genomic.txt | parallel --jobs $STEP bash $WD/find_orfs.sh
cd $WD
