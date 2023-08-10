Setting up data
===============
You must install the [deep-taxon repository](https://github.com/exabiome/deep-taxon). Its recommended to 
first create a fresh conda environment for installing this.


```bash
git checkout git@github.com:exabiome/deep-taxon.git deep-taxon.git
cd deep-taxon.git
conda env create --name deep-taxon --file=environments.yml
conda activate deep-taxon
python setup.py install
```

Get test accessions from metadata file and make file-of-files
```bash
deep-taxon get-accessions -t ../../input/gtdb/r207/metadata_r207.tsv > metadata_r207.test.accs.tsv
deep-taxon make-fof --genomic $CFS/m3513/endurable/ncbi/genomes metadata_r207.test.accs.tsv > metadata_r207.test.fof
```

Load balance by splitting files by size

```bash
deep-taxon split-fof metadata_r207.test.fof 2000 contigs/metadata_r207.test > contigs/split_file.log 
```

Append accessions to contig names for post-processing of predictions
```bash
sbatch append_all_chunks.sh
```

This script should be backed up with this file, but in case it is not, here is what it should look like:
```bash
#!/bin/bash
#SBATCH -J append
#SBATCH -A m2865
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2
#SBATCH -n 1
#SBATCH -o contigs/slurm/append.%A_%a.log
#SBATCH -e contigs/slurm/append.%A_%a.log
#SBATCH -n 1
#SBATCH --array=0-1999

module load parallel
date +%s
i=`printf "%04d\n" $SLURM_ARRAY_TASK_ID`
fof=contigs/metadata_r207.test.$i.fof
out=contigs/metadata_r207.test.$i.fna
echo $fof $out
cat $fof | parallel -j 128 "deep-taxon append-accessions {}" > $out
date +%s
```


Make bin directories for classifying bins

```bash
cd bins
ls ../contigs/metadata_r207.test.*.fof | parallel -j 50 bash make_dir.sh > make_dirs.log 2>&1 &
```

`make_dir.sh` should be backed up with this file, but in case it is not, here is what it should look like:

```bash
fof=$1
dir=`basename $fof | cut -f3 -d.`
mkdir $dir
for f in `cat $fof`; do
    zcat $f > $dir/`basename $f .gz`
done
```

Running CAT
===========
Create and activate the `cat-gtnet-paper` environment:

```bash
conda create --file=cat-gtnet-paper.yaml
conda activate cat-gtnet-paper
```

Run from `./cat` directory and activate `cat-gtnet-paper` environment

```bash
conda activate cat-gtnet-paper
```

Contigs
-------
Run CAT on chunks created in "Settup up data" above. Do this using a Slurm Job Array

```bash
sbatch cmd_contigs.array.sh
```

Bins
----
Run CAT on bins created in "Settup up data" above. Do this using a Slurm Job Array

```bash
sbatch cmd_bins.array.sh
```


Running Sourmash
================
Create and activate the `sourmash-gtnet-paper` environment:

```bash
conda create --file=sourmash-gtnet-paper.yaml
conda activate sourmash-gtnet-paper
```

Run from `./sourmash` and activate `sourmash-gtnet-paper` environment

Contigs
-------
Run Sourmash on chunks created in "Settup up data" above. Do this using a Slurm Job Array

```bash
sbatch cmd_contigs.array.sh
```

This relies on the script `cmd_contigs.sh`. It should be backed up with this file, but in case it is not, here is what it should look like:

```bash
i=${1:?"Missing partition argument"}

K="51"
INDIR="../contigs"
DB="gtdb-rs207.genomic-reps.dna.k$K.lca.json.gz"
FNA="$INDIR/metadata_r207.test.$i.fna"
SIG="$INDIR/metadata_r207.test.$i.sig"
CSV="$INDIR/metadata_r207.test.$i.sourmash.csv"

sourmash sketch dna $FNA --singleton -p scaled=10000,k=$K -o $SIG
sourmash lca classify --db $DB --query $SIG -o $CSV
```

Bins
----
Run Sourmash on chunks created in "Settup up data" above. Do this using a Slurm Job Array

```bash
sbatch cmd_bins.array.sh
```

This relies on the script `cmd_bins.sh`. It should be backed up with this file, but in case it is not, here is what it should look like:

```bash
i=${1:?"Missing partition argument"}

K="51"
INDIR="../bins"
DB="gtdb-rs207.genomic-reps.dna.k$K.lca.json.gz"
FNA_DIR="$INDIR/$i"
SIG="$INDIR/metadata_r207.test.$i.sig"
CSV="$INDIR/metadata_r207.test.$i.sourmash.csv"

sourmash sketch dna $FNA_DIR/*.fna -p scaled=10000,k=51 -o $SIG
sourmash lca classify --db $DB --query $SIG -o $CSV
```
