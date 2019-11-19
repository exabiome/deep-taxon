import math

import numpy as np

from hdmf.common import VectorIndex, VectorData, DynamicTable,\
                        register_class, load_namespaces
from hdmf.utils import docval, call_docval_func, get_docval, popargs
from hdmf import Container


NS = 'deep-index'

@register_class('BitpackedIndex', NS)
class BitpackedIndex(VectorIndex):

    def __get_single_item(self, i):
        start = 0 if i == 0 else self.data[i-1]
        end = self.data[i]
        shift = start % 2
        unpacked = np.unpackbits(self.target[start//2:math.ceil(end/2)])
        unpacked = unpacked.reshape(-1, 4)[shift:shift+end-start].T
        return unpacked

    def __getitem__(self, args):
        """
        Slice ragged array of *packed* one-hot encoded DNA sequence
        """
        return self.__get_single_item(args[0])


@register_class('DNATable', NS)
class DNATable(DynamicTable):

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'names', 'type': ('array_data', 'data'), 'doc': 'sequence names'},
            {'name': 'sequence', 'type': ('array_data', 'data'), 'doc': 'bitpacked DNA sequence'},
            {'name': 'index', 'type': ('array_data', 'data'), 'doc': 'index for sequence'},
            {'name': 'taxon', 'type': ('array_data', 'data'), 'doc': 'index for sequence'})
    def __init__(self, **kwargs):
        names, index, sequence, taxon = popargs('names', 'index', 'sequence', 'taxon', kwargs)
        columns = list()
        columns.append(VectorData('names', 'sequence names', data=names))
        columns.append(VectorData('sequence', 'bitpacked DNA sequences', data=sequence))
        columns.append(BitpackedIndex('sequence_index', index, columns[-1]))
        columns.append(VectorData('taxon', 'taxa for each sequence', data=taxon))
        kwargs['columns'] = columns
        call_docval_func(super().__init__, kwargs)


@register_class('TaxaTable', NS)
class TaxaTable(DynamicTable):

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'taxon_id', 'type': ('array_data', 'data'), 'doc': 'the taxon ID'},
            {'name': 'embedding', 'type': ('array_data', 'data'), 'doc': 'the embedding for each taxon'})
    def __init__(self, **kwargs):
        taxon_id, embedding = popargs('taxon_id', 'embedding', kwargs)
        columns = list()
        columns.append(VectorData('taxon_id', 'taxonomy IDs from NCBI', data=taxon_id))
        columns.append(VectorData('embedding', 'an embedding for each species', data=embedding))
        kwargs['columns'] = columns
        call_docval_func(super().__init__, kwargs)

@register_class('DeepIndexFile', NS)
class DeepIndexFile(Container):

    __fields__ = ('dna_table', 'taxa_table')

    @docval({'name': 'dna_table', 'type': DNATable, 'doc': 'the table storing DNA sequences'},
            {'name': 'taxa_table', 'type': TaxaTable, 'doc': 'the table storing taxa information'})
    def __init__(self, **kwargs):
        dna_table, taxa_table = popargs('dna_table', 'taxa_table', kwargs)
        call_docval_func(super().__init__, {'name': 'root'})
        self.dna_table, self.taxa_table = dna_table, taxa_table
