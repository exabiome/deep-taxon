i=${1:?"Missing partition argument"}

K="51"
INDIR="../bins"
DB="gtdb-rs207.genomic-reps.dna.k$K.lca.json.gz"
FNA_DIR="$INDIR/$i"
SIG="$INDIR/metadata_r207.test.$i.sig"
CSV="$INDIR/metadata_r207.test.$i.sourmash.csv"

sourmash sketch dna $FNA_DIR/*.fna -p scaled=10000,k=51 -o $SIG
sourmash lca classify --db $DB --query $SIG -o $CSV
