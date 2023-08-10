import argparse
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score

from deep_taxon.utils import parse_logger

parser = argparse.ArgumentParser()
parser.add_argument('lca', type=str, help='sourmash LCA output')
parser.add_argument('metadata', type=str, help='GTDB metadata file')
parser.add_argument('-o', '--output', type=str, help='the output file to save results to', default=None)

args = parser.parse_args()


"""
ID,status,superkingdom,phylum,class,order,family,genus,species,strain
/global/cfs/cdirs/m3513/genomics/ncbi/genomes/all/GCF/003/367/205/GCF_003367205.1_ASM336720v1/GCF_003367205.1_ASM336720v1_genomic.fna.gz,found,d__Bacteria,p__Actinobacteriota,c__Actinomycetia,o__Actinomycetales,f__Micrococcaceae,g__Pseudarthrobacter,s__Pseudarthrobacter sp900168295,
/global/cfs/cdirs/m3513/genomics/ncbi/genomes/all/GCA/902/617/415/GCA_902617415.1_AG-892-F15/GCA_902617415.1_AG-892-F15_genomic.fna.gz,found,d__Bacteria,p__Proteobacteria,c__Alphaproteobacteria,o__Pelagibacterales,f__Pelagibacteraceae,g__Pelagibacter,,
/global/cfs/cdirs/m3513/genomics/ncbi/genomes/all/GCF/006/716/625/GCF_006716625.1_ASM671662v1/GCF_006716625.1_ASM671662v1_genomic.fna.gz,found,d__Bacteria,p__Firmicutes,c__Bacilli,o__Mycoplasmatales,f__Mycoplasmoidaceae,g__Ureaplasma,s__Ureaplasma urealyticum,
"""

logger = parse_logger('')

taxlevels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
# extra_cols = ['contig_count', 'checkm_completeness']
def func(row):
    dat = dict(zip(taxlevels, row['gtdb_taxonomy'].split(';')))
    dat['species'] = dat['species'] # .split(' ')[1]
    dat['gtdb_genome_representative'] = row['gtdb_genome_representative'][3:]
    dat['accession'] = row['accession'][3:]
    # for k in extra_cols:
    #     dat[k] = row[k]
    return pd.Series(data=dat)

# keep_cols = ['accession', 'gtdb_taxonomy', 'gtdb_genome_representative', 'contig_count', 'checkm_completeness']

logger.info(f'reading GTDB metadata file from {args.metadata}')
keep_cols = ['accession', 'gtdb_taxonomy', 'gtdb_genome_representative']
taxdf = pd.read_csv(args.metadata, header=0, sep='\t', usecols=keep_cols).apply(func, axis=1).set_index('accession')


# "GCA_000380905.1-1094-AQYW01000001.1 Nanoarchaeota archaeon SCGC AAA011-L22 DUSEL4_1DRAFT_contig_80.81_C, whole genome shotgun sequence",nomatch,,,,,,,,
def func(row):
    ID = row['ID']
    ar = ID.split('-')
    row['accession'] = ar[0]
    row['length'] = int(ar[1])
    row['ID'] = ar[2]
    return row

logger.info(f'reading sourmash LCA taxonomy file from {args.lca}')
lca_df = pd.read_csv(args.lca).apply(func, axis=1).set_index('accession')

taxdf = taxdf.filter(lca_df.index, axis=0)

results = {'accuracy': list(), 'pclfd': list(), 'bases_pclfd': list()}
for col in taxlevels[1:]:
    logger.info(f'computing results for {col}')
    mask = lca_df[col].notna()
    results['pclfd'].append(mask.mean())
    length = lca_df['length']
    results['bases_pclfd'].append(length[mask].sum()/length.sum())
    true = taxdf[col][mask]
    pred = lca_df[col][mask]
    results['accuracy'].append(accuracy_score(true, pred))


res_df = pd.DataFrame(data=results, index=taxlevels[1:])
print(res_df)

if args.output is not None:
    logger.info(f'saving results to {args.output}')
    res_df.to_csv(args.output)
