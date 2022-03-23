#!/bin/bash
#SBATCH --qos=regular
#SBATCH -A m3513
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=270
#SBATCH -o orf_lca.parallel.log
#SBATCH -e orf_lca.parallel.log
#SBATCH -J orf_lca

module load parallel
conda activate cat

WD="$SCRATCH/exabiome/deep-taxon/sota/diamond"
SCRIPT="$WD/orf_lca.sh"
cd $WD

ls $WD/nonrep_blastp/r202.nonrep.blastp*[0-9] | parallel --jobs 12 $SCRIPT {}
