## Phylogenetic embedding
- Phylogenetic distances were obtained from phylogenetic trees obtained from the GTDB (Genome Taxonomy Database) 
  - https://data.ace.uq.edu.au/public/gtdb/data/releases/release89/89.0/
  - this provides NCBI Genbank/Refseq accessions
  - ar122_r89.tree and ac120_r89.tree were used
    - internode labels were removed because my Newick parser cannot handle them
  - distances were calculated using tree2dmat
    - code is available in ../wgsim.git/tree2dmat.c
  - distances were used by MDS to embed taxa into latent space
    - code is available in mds.py

## Genomic sequence
- Genomes included in the GTDB trees were retrieved from the NCBI FTP site using rsync
  - ftp://ftp.ncbi.nlm.nih.gov/genomes/all/
  - For genomes that did not have CDS, ORFfinder was used to call CDS
    - ORFfinder was obtained from NCBI
      - ftp://ftp.ncbi.nlm.nih.gov/genomes/TOOLS/ORFfinder/linux-i64/ORFfinder.gz
    - ORFfinder was called with the following parameters
      - ORFfinder -in $fna -out $cdsout -n true -outfmt 1 
