#!/bin/bash
#SBATCH -J cat_ctgs
#SBATCH -A m2865
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 70
#SBATCH -o /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/contigs/cat_log/%A_%a.log
#SBATCH -e /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/contigs/cat_log/%A_%a.log
#SBATCH -n 1
#SBATCH --array=1-1999

i=`printf "%04d\n" $SLURM_ARRAY_TASK_ID`

CHUNK_DIR="/pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/contigs"
BASE="$CHUNK_DIR/metadata_r207.test.$i"
FNA="$BASE.fna"
OUT_PFX="$BASE.CAT"

echo "T_START `date +%s`"
CAT.git/CAT_pack/CAT contigs --force \
--contigs_fasta $FNA \
--database_folder gtdb_prepared/db \
--taxonomy_folder gtdb_prepared/tax \
--out_prefix $OUT_PFX
echo "T_END `date +%s`"
