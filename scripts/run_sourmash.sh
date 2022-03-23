#!/bin/bash
#SBATCH --qos=genepool
#SBATCH -A genother
#SBATCH --nodes=1
#SBATCH --time=10
#SBATCH -o benchmarks/run_sourmash.%j.log
#SBATCH -e benchmarks/run_sourmash.%j.log
#SBATCH -J sourmash_pipeline
# SBATCH --constraint=haswell

############################
# This script expects the following files and directory structure:
#
# ./
#   run_sourmash.sh                        # This script
#
# benchmarks/
#   r202.nonrep.sample.files.small.txt      # a file of Fasta file paths for running the benchmark. Files should be named using NCBI
#                                     # convention for naming files from NCBI Assembly
#   append_accession.py               # a script for appending NCBI accessions to sequences in Fasta file
#
# sourmash/
#   db/gtdb-rs202.genomic-reps.k31.lca.json.gz
#

function log() {
    echo "`date +%Y-%m-%d\ %T,%3N` - RUN_CAT - $1"
}

DIR="benchmarks/run_sourmash.$SLURM_JOB_ID"
mkdir -p $DIR
log "storing results in $DIR"

FASTA_FOF="benchmarks/r202.nonrep.sample.files.small.txt"
ALL_FNA="$DIR/all_genomic.fna"
ACC_SCRIPT="benchmarks/append_accession.py"

log "workflow_begin `date +%s`"
log "concatenating fastas and adding labels"
echo -n > $ALL_FNA
for FA in `sourmash $FASTA_FOF`; do
    echo $FA
    python $ACC_SCRIPT $FA >> $ALL_FNA
done

ALL_SIG="$DIR/all_genomic.sig"

log "sourmash_begin `date +%s`"
log "sketch_begin `date +%s`"
sourmash sketch dna $ALL_FNA --singleton -p scaled=10000,k=31 -o $ALL_SIG
log "sketch_end `date +%s`"

DB="sourmash/db/gtdb-rs202.genomic-reps.k31.lca.json.gz"
SEQ_LCA="$DIR/tax_class.csv"
log "classify_begin `date +%s`"
sourmash lca classify --db $DB --query $ALL_SIG -o $SEQ_LCA
log "classify_end `date +%s`"

# Don't include calculating accuracy as part of the timing
log "sourmash_end `date +%s`"
log "workflow_end `date +%s`"

# TAX_ACC="$DIR/tax_acc.csv"
# log "taxacc_begin `date +%s`"
# python $SCRIPT tax-acc -o $TAX_ACC $SEQ_LCA $TAX_CSV $ALL_FNA
# log "taxacc_end  `date +%s`"
# log "workflow_end `date +%s`"
