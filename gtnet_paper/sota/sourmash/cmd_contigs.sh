i=${1:?"Missing partition argument"}

K="51"
INDIR="../contigs"
DB="gtdb-rs207.genomic-reps.dna.k$K.lca.json.gz"
FNA="$INDIR/metadata_r207.test.$i.fna"
SIG="$INDIR/metadata_r207.test.$i.sig"
CSV="$INDIR/metadata_r207.test.$i.sourmash.csv"

sourmash sketch dna $FNA --singleton -p scaled=10000,k=$K -o $SIG
sourmash lca classify --db $DB --query $SIG -o $CSV
