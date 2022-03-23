#!/bin/bash -l
#SBATCH --qos=regular
#SBATCH -A m3513
#SBATCH --constraint=haswell
#SBATCH -J diamond_blastp
#SBATCH --nodes=504
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=2
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=none
#SBATCH --time=06:00:00
#SBATCH -o diamond_blastp.%j.log
#SBATCH -e diamond_blastp.%j.log


module purge
module load gcc impi
conda activate cat
export SLURM_HINT=multithread


DIR="$SCRATCH/exabiome/deep-taxon/sota/diamond"
TMPDIR="$DIR/tmpdir"
PTMPDIR="$DIR/ptmpdir"
DB="$DIR/r202.rep.protein"
REF="$DIR/r202.rep.protein.faa.gz"

QUERY="$DIR/r202.nonrep.transorf.faa.gz"
OUTPUT="$DIR/nonrep_blastp/r202.nonrep.blastp"

#QUERY="$DIR/test.transorf.faa.gz"
#OUTPUT="$DIR/test.blastp.tsv"

FLAGS="blastp --db $DB.dmnd --query $QUERY -f 6 -o $OUTPUT --multiprocessing --tmpdir $TMPDIR --parallel-tmpdir $PTMPDIR"
date
srun diamond $FLAGS > diamond_blastp.log 2>&1
date
