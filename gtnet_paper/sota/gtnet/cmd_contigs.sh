arg=$((SLURM_LOCALID + 1))
part=${!arg}

INDIR="../contigs"
FNA="$INDIR/metadata_r207.test.$part.fna"
RAW="$INDIR/metadata_r207.test.$part.gtnet.raw.csv"
CSV="$INDIR/metadata_r207.test.$part.gtnet.csv"

gtnet predict -s -g -D $SLURM_LOCALID -o $RAW $FNA
gtnet filter -o $CSV $RAW
