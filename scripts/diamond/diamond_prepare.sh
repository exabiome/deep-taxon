conda activate cat

DIR="$SCRATCH/exabiome/deep-taxon/sota/diamond"
TMPDIR="$DIR/tmpdir"
PTMPDIR="$DIR/ptmpdir"
DB="$DIR/r202.rep.protein"
REF="$DIR/r202.rep.protein.faa.gz"
QUERY="$DIR/r202.nonrep.transorf.faa.gz"

mkdir -p $TMPDIR
mkdir -p $PTMPDIR

echo "`date` makedb start"
diamond makedb -p 10 --db $DB --in $REF > makedb.log 2>&1 
echo "`date` makedb end"

echo "`date` blastp start"
diamond blastp -p 10 --db $DB --query $QUERY --multiprocessing --mp-init --tmpdir $TMPDIR --parallel-tmpdir $PTMPDIR > blastp.prepare.log 2>&1 
echo "`date` blastp end"
