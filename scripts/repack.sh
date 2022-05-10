conda activate deep-taxon-3
$IN=${1:?'Please provide input Deep Taxon file'}
$OUT=${2:?'Please provide output file to save decompressed data to'}
h5repack -v -f /seq_table/sequence:NONE $IN $OUT
