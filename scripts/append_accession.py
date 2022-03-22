import argparse
import os
import sys

from skbio import DNA
import skbio.io as skio

desc = "fasta_path must be a file path from NCBI and have the genome assembly accession in it"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('fasta_path', type=str, help='the fasta file to append the prefix to')

args = parser.parse_args()
args.prefix = os.path.basename(args.fasta_path)[:15]

seqs = list()
tmp_fa = sys.stdout
w = 100
for seq in skio.read(args.fasta_path, format='fasta', constructor=DNA):
    seq.metadata['id'] = args.prefix+"-"+str(len(seq))+"-"+seq.metadata['id']
    seqs.append(seq)
    tmp_fa.write('>')
    tmp_fa.write(seq.metadata['id'])
    tmp_fa.write('\n')
    for s in range(0, len(seq), w):
        tmp_fa.write(''.join(seq.values[s:s+w].astype('U')))
        tmp_fa.write('\n')
