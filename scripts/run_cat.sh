#!/bin/bash
#SBATCH --qos=regular
#SBATCH -A m3513
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=1440
#SBATCH -o benchmarks/run_cat.%j.log
#SBATCH -e benchmarks/run_cat.%j.log
#SBATCH -J cat_pipeline

############################
# This script expects the following files and directory structure:
#
# ./
#   run_cat.sh                        # This script
#
# benchmarks/
#   r202.nonrep.sample.files.txt      # a file of Fasta file paths for running the benchmark. Files should be named using NCBI
#                                     # convention for naming files from NCBI Assembly
#   append_accession.py              # a script for appending NCBI accessions to sequences in Fasta file
#
# diamond/
#   r202.rep.protein.dmnd             # the DIAMOND database made from GTDB representative proteins. Fasta headers for 
#                                     # sequences used to make the database should have their NCBI accession appended to them
#   blast_lca.py                      # a script for running LCA on DIAMOND outputs and making taxonomic classications for each sequence
#

function log() {
    echo "`date +%Y-%m-%d\ %T,%3N` - RUN_CAT - $1"
}

DIR="benchmarks/run_cat.$SLURM_JOB_ID"
mkdir -p $DIR
log "storing results in $DIR"

FASTA_FOF="benchmarks/r202.nonrep.sample.files.txt"
ALL_FNA="$DIR/all_genomic.fna"
ACC_SCRIPT="benchmarks/append_accession.py"

log "workflow_begin `date +%s`"
log "concatenating fastas and adding labels"
echo "" > $ALL_FNA
for FA in `cat $FASTA_FOF`; do
    echo $FA
    python $ACC_SCRIPT $FA >> $ALL_FNA
done

ALL_FAA="$DIR/all_genomic.proteins.faa"
CMD="prodigal -a $ALL_FAA"

log "cat_begin `date +%s`"
log "prodigal_begin `date +%s`"
cat $ALL_FNA | $CMD > $DIR/prodigal.log 2>&1 
log "prodigal_end `date +%s`"

DB="diamond/r202.rep.protein.dmnd"
DIAMOND_OUTPUT="$DIR/diamond.out"
FLAGS="blastp --db $DB --query $ALL_FAA -f 6 -o $DIAMOND_OUTPUT --tmpdir $TMPDIR"
log "diamond_begin `date +%s`"
diamond $FLAGS > $DIR/diamond.log 2>&1 
log "diamond_end `date +%s`"

SCRIPT="diamond/blast_lca.py"
TAX_CSV="r202_taxonomy.csv"
ORF_LCA="$DIR/orf-lca.csv"
log "orflca_begin `date +%s`"
python $SCRIPT orf-lca -t $TAX_CSV $DIAMOND_OUTPUT $ORF_LCA
log "orflca_end `date +%s`"

SEQ_LCA="$DIR/tax_class.csv"
log "aggorfs_begin `date +%s`"
python $SCRIPT agg-orfs -o $SEQ_LCA $ORF_LCA
log "aggorfs_end  `date +%s`"

# Don't include calculating accuracy as part of the timing
log "cat_end `date +%s`"

TAX_ACC="$DIR/tax_acc.csv"
log "taxacc_begin `date +%s`"
python $SCRIPT tax-acc -o $TAX_ACC $SEQ_LCA $TAX_CSV
log "taxacc_end  `date +%s`"
log "workflow_end `date +%s`"
