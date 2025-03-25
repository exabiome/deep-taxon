#!/bin/bash
#SBATCH -J cat_bins
#SBATCH -A m2865
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 70
#SBATCH -o /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/bins/cat_log/%A_%a.log
#SBATCH -e /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/bins/cat_log/%A_%a.log
#SBATCH -n 1
#SBATCH --array=1-1999

i=`printf "%04d\n" $SLURM_ARRAY_TASK_ID`

BINS_DIR="/pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/bins"
BIN_DIR="$BINS_DIR/$i"
OUT_PFX="$BINS_DIR/metadata_r207.test.$i.BAT"

echo "T_START `date +%s`"
CAT.git/CAT_pack/CAT bins --force \
--bin_folder $BIN_DIR \
--database_folder gtdb_prepared/db \
--taxonomy_folder gtdb_prepared/tax \
--out_prefix $OUT_PFX
echo "T_END `date +%s`"
