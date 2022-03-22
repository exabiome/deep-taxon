#!/bin/bash
#SBATCH --qos=regular
#SBATCH -A m2865
#SBATCH --nodes=1
#SBATCH --time=60
#SBATCH -o benchmarks/run_gtnet.%j.log
#SBATCH -e benchmarks/run_gtnet.%j.log
#SBATCH -J gput_gtnet_pipeline
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=8

############################
# This script expects the following files and directory structure:
#
# ./
#   run_gtnet.sh                        # This script
#   r202_taxonomy.csv                 # The GTDB taxonomy file from running diamond/blast_lca.py prep-meta
#   gtnet.deploy                      # The GTNet deployment directory
#
# benchmarks/
#   r202.nonrep.sample.files.small.txt      # a file of Fasta file paths for running the benchmark. Files should be named using NCBI
#                                     # convention for naming files from NCBI Assembly
#   append_accession.py               # a script for appending NCBI accessions to sequences in Fasta file
#

function log() {
    echo "`date +%Y-%m-%d\ %T,%3N` - RUN_CAT - $1"
}

DIR="benchmarks/run_gtnet.$SLURM_JOB_ID"
mkdir -p $DIR
log "storing results in $DIR"

FASTA_FOF="benchmarks/r202.nonrep.sample.files.small.txt"
#FASTA_FOF="benchmarks/r202.nonrep.sample.files.txt"
ALL_FNA="$DIR/all_genomic.fna"
ACC_SCRIPT="benchmarks/append_accession.py"

log "workflow_begin `date +%s`"
log "concatenating fastas and adding labels"
echo -n > $ALL_FNA
for FA in `cat $FASTA_FOF`; do
    echo $FA
    python $ACC_SCRIPT $FA >> $ALL_FNA
done

DEPLOY_DIR="gtnet.deploy"
TAX_CSV="r202_taxonomy.csv"
SEQ_LCA="$DIR/tax_class.csv"
log "gtnet_begin `date +%s`"
echo "deep-taxon onnx-run -o $SEQ_LCA $DEPLOY_DIR $ALL_FNA"
srun deep-taxon onnx-run -o $SEQ_LCA $DEPLOY_DIR $ALL_FNA
log "gtnet_end `date +%s`"
log "workflow_end `date +%s`"

# TAX_ACC="$DIR/tax_acc.csv"
# log "taxacc_begin `date +%s`"
# python $SCRIPT tax-acc -o $TAX_ACC $SEQ_LCA $TAX_CSV $ALL_FNA
# log "taxacc_end  `date +%s`"
# log "workflow_end `date +%s`"
