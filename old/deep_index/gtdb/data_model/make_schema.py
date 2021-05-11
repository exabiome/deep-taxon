from hdmf.spec import GroupSpec, DatasetSpec, NamespaceBuilder

# create a builder for the namespace
ns_builder = NamespaceBuilder("Model for storing data to be read in for deep-index training", "deep-index")

bpvector = DatasetSpec('A VectorIndex that indexes bitpacked DNA',
                       dtype='int',
                       data_type_inc='VectorIndex',
                       data_type_def='BitpackedIndex')

datasets = [
    DatasetSpec('Names for sequences', name='names', dtype='text', data_type_inc='VectorData'),
    DatasetSpec('Packed DNA sequence', name='sequence', dtype='uint8', data_type_inc='VectorData'),
    DatasetSpec("Index for 'sequence'", name='sequence_index', data_type_inc='BitpackedIndex'),
]

table = GroupSpec('A table for storing sequence data',
                  datasets=datasets,
                  data_type_inc='DynamicTable',
                  data_type_def='DNATable')

spec_src = "dna_table.yaml"
ns_builder.add_spec(spec_src, bpvector)
ns_builder.add_spec(spec_src, table)
ns_builder.include_namespace('hdmf-common')

ns_filepath = "deep_index.namespace.yaml"
ns_builder.export(ns_filepath)

