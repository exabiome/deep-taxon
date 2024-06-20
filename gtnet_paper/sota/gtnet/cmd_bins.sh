part=${1:?"Missing partition argument"}

INDIR="../bins"
FNA_DIR="$INDIR/$part"
CSV="$INDIR/metadata_r207.test.$part.gtnet.$SLURM_LOCALID.csv"

gtnet classify -g -D $SLURM_LOCALID -o $CSV `ls $FNA_DIR/*.fna | awk "NR % 4 == $SLURM_LOCALID"`
