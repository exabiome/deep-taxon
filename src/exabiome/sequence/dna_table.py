from abc import ABCMeta, abstractmethod
import math

import numpy as np
import torch

from hdmf.common import VectorIndex, VectorData, DynamicTable,\
                        DynamicTableRegion, register_class
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
            return self._get_single_item(args)
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
        packed = self.target[start:end]
        bits = np.unpackbits(packed)
        bits = bits[bits.shape[0] % 5:]
        bits = bits.reshape(-1, 5)
        unpacked = np.pad(bits, ((0, 0), (3, 0)), mode='constant', constant_values=((0, 0), (0, 0)))
        ohe_pos = np.packbits(unpacked, axis=1).squeeze()
        if ohe_pos[0] == 0:
            ohe_pos = ohe_pos[1:]
        ohe_pos = ohe_pos - 1
        return ohe_pos


class SequenceTable(DynamicTable, metaclass=ABCMeta):

    @abstractmethod
    def get_index(self, data, target):
        pass

    @abstractmethod
    def get_torch_conversion(self, dtype=None, device=None):
        pass

    @abstractmethod
    def get_numpy_conversion(self):
        pass

    def set_torch(self, use_torch, dtype=None, device=None):
        if use_torch:
            self.convert = self.get_torch_conversion(dtype, device)
        else:
            self.convert = self.get_numpy_conversion()

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'names', 'type': ('array_data', 'data', VectorData), 'doc': 'sequence names'},
            {'name': 'sequence', 'type': ('array_data', 'data', VectorData), 'doc': 'bitpacked DNA sequence'},
            {'name': 'sequence_index', 'type': ('array_data', 'data', BitpackedIndex), 'doc': 'index for sequence'},
            {'name': 'taxon', 'type': ('array_data', 'data', VectorData), 'doc': 'index for sequence'},
            {'name': 'taxon_table', 'type': DynamicTable, 'doc': "target for 'taxon'", 'default': None})
    def __init__(self, **kwargs):
        names, index, sequence, taxon, taxon_table = popargs('names',
                                                             'sequence_index',
                                                             'sequence',
                                                             'taxon',
                                                             'taxon_table',
                                                             kwargs)
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
        self.convert = self.get_numpy_conversion()

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        else:
            ret = list(super().__getitem__(key))
            # sequence data will come from the third column
            ret[2] = self.convert(ret[2])
            return tuple(ret)


@register_class('DNATable', NS)
class DNATable(SequenceTable):

    def get_torch_conversion(self, dtype=None, device=None):
        return lambda x: torch.as_tensor(x, dtype=dtype, device=device).T

    def get_numpy_conversion(self):
        return lambda x: x

    def get_index(self, data, target):
        return PackedDNAIndex('sequence_index', data, target)


@register_class('AATable', NS)
class AATable(SequenceTable):

    charmap = np.array(['A', 'B',
                        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '', '', '',
                        '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                        '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                       dtype='<U1')

    def get_numpy_conversion(self):
        def func(x):
            ret = np.zeros([x.shape[0], 26], dtype=float)
            ret[np.arange(ret.shape[0]), x] = 1.0
            return ret
        return func

    def get_torch_conversion(self, dtype=None, device=None):
        def func(x):
            ret = torch.zeros([x.shape[0], 26], dtype=dtype, device=device)
            ret[np.arange(ret.shape[0]), x.tolist()] = 1.0
            ret = ret.T
            return ret
        return func

    def get_index(self, data, target):
        return PackedAAIndex('sequence_index', data, target)

    @classmethod
    def to_sequence(self, data):
        return "".join(self.charmap[i] for i in np.where(data)[1])

    def get(self, idx, sequence=False):
        ret = list(self[idx])
        if sequence:
            ret[2] = self.to_sequence(ret[2])
        return ret


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
        (seq_i, seq_name, sequence, (tax_i, taxon_name, taxon_emb)) = self.seq_table[i]
        return {'taxon': taxon_name, 'name': seq_name, "sequence": sequence, "embedding": taxon_emb}

    def __len__(self):
        return len(self.seq_table)

    def set_torch(self, use_torch, dtype=None, device=None):
        self.seq_table.set_torch(use_torch, dtype=dtype, device=device)

    def to_sequence(self, data):
        return self.seq_table.to_sequence(data)
