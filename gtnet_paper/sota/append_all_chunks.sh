#!/bin/bash
#SBATCH -J append
#SBATCH -A m2865
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2
#SBATCH -n 1
#SBATCH -o contigs/slurm/append.%A_%a.log
#SBATCH -e contigs/slurm/append.%A_%a.log
#SBATCH -n 1
#SBATCH --array=0-1999

module load parallel
date +%s
i=`printf "%04d\n" $SLURM_ARRAY_TASK_ID`
fof=contigs/metadata_r207.test.$i.fof
out=contigs/metadata_r207.test.$i.fna
echo $fof $out
cat $fof | parallel -j 128 "deep-taxon append-accessions {}" > $out
date +%s
