conda activate cat
HITS=$1
TAX="$SCRATCH/exabiome/deep-taxon/input/gtdb/r202/r202_taxonomy.csv"
SCRIPT="$SCRATCH/exabiome/deep-taxon/sota/diamond/blast_lca.py"
echo "Processing $HITS"
python $SCRIPT orf-lca -t $TAX $HITS $HITS.orf_lca.csv > $HITS.log 2>&1
