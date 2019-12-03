from abc import ABCMeta, abstractmethod
import math

import numpy as np

from hdmf.common import VectorIndex, VectorData, DynamicTable,\
                        DynamicTableRegion, register_class, load_namespaces
from hdmf.utils import docval, call_docval_func, get_docval, popargs
from hdmf import Container, Data


NS = 'deep-index'

class BitpackedIndex(VectorIndex, metaclass=ABCMeta):


    def _start_end(self, i):
        start = 0 if i == 0 else self.data[i-1]
        end = self.data[i]
        return start, end

    @abstractmethod
    def _get_single_item(self, i):
        pass

    def __getitem__(self, args):
        """
        Slice ragged array of *packed* one-hot encoded DNA sequence
        """
        if isinstance(args, int):
            return self.__get_single_item(args)
        else:
            raise ValueError("Can only index bitpacked sequence with integers")


@register_class('PackedDNAIndex', NS)
class PackedDNAIndex(BitpackedIndex):

    def _get_single_item(self, i):
        start, end = self._start_end(i)
        shift = start % 2
        unpacked = np.unpackbits(self.target[start//2:math.ceil(end/2)])
        unpacked = unpacked.reshape(-1, 4)[shift:shift+end-start].T
        return unpacked


@register_class('PackedAAIndex', NS)
class PackedAAIndex(BitpackedIndex):

    def _get_single_item(self, i):
        start, end = self._start_end(i)
        shift = start % 2
        unpacked = np.unpackbits(self.target[start:end]).reshape(-1, 8)
        unpacked = unpacked[:, ]
        return unpacked


class SequenceTable(DynamicTable, metaclass=ABCMeta):

    @abstractmethod
    def get_index(self, data, target):
        pass

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'names', 'type': ('array_data', 'data', VectorData), 'doc': 'sequence names'},
            {'name': 'sequence', 'type': ('array_data', 'data', VectorData), 'doc': 'bitpacked DNA sequence'},
            {'name': 'sequence_index', 'type': ('array_data', 'data', BitpackedIndex), 'doc': 'index for sequence'},
            {'name': 'taxon', 'type': ('array_data', 'data', VectorData), 'doc': 'index for sequence'},
            {'name': 'taxon_table', 'type': DynamicTable, 'doc': "target for 'taxon'", 'default': None})
    def __init__(self, **kwargs):
        names, index, sequence, taxon, taxon_table = popargs('names', 'sequence_index', 'sequence', 'taxon', 'taxon_table', kwargs)
        columns = list()
        if isinstance(names, VectorData):      # data is being read -- here we assume that everything is a VectorData
            columns.append(names)
            columns.append(index)
            columns.append(sequence)
            columns.append(taxon)
        else:
            columns.append(VectorData('names', 'sequence names', data=names))
            columns.append(VectorData('sequence', 'bitpacked DNA sequences', data=sequence))
            columns.append(self.get_index(index, columns[-1]))
            columns.append(DynamicTableRegion('taxon', taxon, 'taxa for each sequence', taxon_table))
        kwargs['columns'] = columns
        call_docval_func(super().__init__, kwargs)


@register_class('DNATable', NS)
class DNATable(SequenceTable):

    def get_index(self, data, target):
        return PackedDNAIndex('sequence_index', data, target)


@register_class('AATable', NS)
class AATable(SequenceTable):

    def get_index(self, data, target):
        return PackedAAIndex('sequence_index', data, target)


@register_class('TaxaTable', NS)
class TaxaTable(DynamicTable):

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'taxon_id', 'type': ('array_data', 'data', VectorData), 'doc': 'the taxon ID'},
            {'name': 'embedding', 'type': ('array_data', 'data', VectorData), 'doc': 'the embedding for each taxon'})
    def __init__(self, **kwargs):
        taxon_id, embedding = popargs('taxon_id', 'embedding', kwargs)
        columns = list()
        if isinstance(taxon_id, VectorData):      # data is being read -- here we assume that everything is a VectorData
            columns.append(taxon_id)
            columns.append(embedding)
        else:
            columns.append(VectorData('taxon_id', 'taxonomy IDs from NCBI', data=taxon_id))
            columns.append(VectorData('embedding', 'an embedding for each species', data=embedding))
        kwargs['columns'] = columns
        call_docval_func(super().__init__, kwargs)


@register_class('CondensedDistanceMatrix', NS)
class CondensedDistanceMatrix(Data):
    pass


@register_class('NewickString', NS)
class NewickString(Data):
    pass


@register_class('DeepIndexFile', NS)
class DeepIndexFile(Container):

    __fields__ = ({'name': 'seq_table', 'child': True},
                  {'name': 'taxa_table', 'child': True},
                  {'name': 'distances', 'child': True},
                  {'name': 'tree', 'child': True})

    @docval({'name': 'seq_table', 'type': (AATable, DNATable), 'doc': 'the table storing DNA sequences'},
            {'name': 'taxa_table', 'type': TaxaTable, 'doc': 'the table storing taxa information'},
            {'name': 'distances', 'type': CondensedDistanceMatrix, 'doc': 'the table storing taxa information'},
            {'name': 'tree', 'type': NewickString, 'doc': 'the table storing taxa information'})
    def __init__(self, **kwargs):
        seq_table, taxa_table, distances, tree = popargs('seq_table', 'taxa_table', 'distances', 'tree', kwargs)
        call_docval_func(super().__init__, {'name': 'root'})
        self.seq_table = seq_table
        self.taxa_table = taxa_table
        self.distances = distances
        self.tree = tree

    def __getitem__(self, i):
        """
        Return a tuple containing (taxon_name, sequence_name, sequence, taxon_embedding)
        """
        (seq_i, seq_name, sequence, (tax_i, taxon_name, taxon_emb)) = self.dna_table[i]
        return {'taxon': taxon_name, 'seqname': seq_name, "sequence": sequence, "embedding": taxon_emb}

    def __len__(self):
        return len(self.dna_table)
