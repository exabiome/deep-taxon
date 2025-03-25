#!/bin/bash
#SBATCH -J gtnet_bins
#SBATCH -A m2865
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 120
#SBATCH -o /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/bins/gtnet_log/%A_%a.log
#SBATCH -e /pscratch/sd/a/ajtritt/exabiome/deep-taxon/gtnet/sota/bins/gtnet_log/%A_%a.log
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 32
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH --array=1-249

for part in $(seq -f "%04g" $SLURM_ARRAY_TASK_ID 250 1999); do
        INDIR="../bins"
        PFX="$INDIR/metadata_r207.test.$part.gtnet"
        srun bash cmd_bins.sh $part

        # merge chunks
        cp $PFX.0.csv $PFX.csv
        grep -h -v ^file $PFX.[1,2,3].csv >> $PFX.csv
        rm $PFX.[0,1,2,3].csv
done
