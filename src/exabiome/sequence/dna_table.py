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
        if np.issubdtype(type(args), np.integer):
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
        ohe_pos = np.packbits(unpacked, axis=1).squeeze(axis=1)
        # trim leading zeros that may be left from padding to
        # ensure enough bits for pack
        # trim trailing zeros in case a non-AA character was
        # used to terminate the original sequence
        ohe_pos = np.trim_zeros(ohe_pos)
        ohe_pos = ohe_pos - 1
        return ohe_pos


class TorchableMixin:

    def get_torch_conversion(self, **kwargs):
        dtype = kwargs.get('dtype')
        device = kwargs.get('device')
        maxlen = kwargs.get('maxlen')
        if np.issubdtype(type(maxlen), np.integer):
            def func(x):
                ret = torch.zeros((maxlen, x.shape[1]))
                ret[0:x.shape[0]] = x
                return ret
        else:
            def func(x):
                return torch.as_tensor(x, dtype=dtype, device=device).T
        return func

    def get_numpy_conversion(self, **kwargs):
        """
        Args:
            maxlen (int):        the maximum sequence length to pad to
        """
        maxlen = kwargs.get('maxlen')
        if np.issubdtype(type(maxlen), np.integer):
            def func(x):
                ret = np.zeros((maxlen, x.shape[1]))
                ret[0:x.shape[0]] = x
                return ret
        else:
            def func(x):
                return x
        return func

    def set_raw(self):
        self.convert = lambda x: x

    def set_torch(self, use_torch, **kwargs):
        """
        Args:
            use_torch :          convert data to torch.Tensors
            maxlen (int):        the maximum sequence length to pad to
        """
        if use_torch:
            self.convert = self.get_torch_conversion(**kwargs)
        else:
            self.convert = self.get_numpy_conversion(**kwargs)


class SequenceTable(DynamicTable, TorchableMixin, metaclass=ABCMeta):

    @abstractmethod
    def get_index(self, data, target):
        pass

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'sequence_name', 'type': ('array_data', 'data', VectorData), 'doc': 'sequence names'},
            {'name': 'sequence', 'type': ('array_data', 'data', VectorData), 'doc': 'bitpacked DNA sequence'},
            {'name': 'sequence_index', 'type': ('array_data', 'data', BitpackedIndex), 'doc': 'index for sequence'},
            {'name': 'length', 'type': ('array_data', 'data', VectorData), 'doc': 'lengths of sequence'},
            {'name': 'taxon', 'type': ('array_data', 'data', VectorData), 'doc': 'index for sequence'},
            {'name': 'taxon_table', 'type': DynamicTable, 'doc': "target for 'taxon'", 'default': None},
            {'name': 'pad', 'type': bool, 'doc': 'pad sequences to the maximum length sequence', 'default': False})
    def __init__(self, **kwargs):
        sequence_name, index, sequence, taxon, taxon_table = popargs('sequence_name',
                                                             'sequence_index',
                                                             'sequence',
                                                             'taxon',
                                                             'taxon_table',
                                                             kwargs)
        self.pad = popargs('pad', kwargs)
        seqlens = popargs('length', kwargs)
        columns = list()
        self.maxlen = None
        if isinstance(sequence_name, VectorData):      # data is being read -- here we assume that everything is a VectorData
            columns.append(sequence_name)
            columns.append(index)
            columns.append(sequence)
            columns.append(seqlens)
            columns.append(taxon)
            if self.pad:   # if we need to pad, compute the maxlen
                self.maxlen = np.max(seqlens.data[:])
        else:
            columns.append(VectorData('sequence_name', 'sequence names', data=sequence_name))
            columns.append(VectorData('sequence', 'bitpacked DNA sequences', data=sequence))
            columns.append(self.get_index(index, columns[-1]))
            columns.append(VectorData('length', 'sequence lengths', data=seqlens))
            columns.append(DynamicTableRegion('taxon', taxon, 'taxa for each sequence', taxon_table))
        kwargs['columns'] = columns
        call_docval_func(super().__init__, kwargs)
        self.convert = self.get_numpy_conversion(maxlen=self.maxlen)

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

    def get_numpy_conversion(self, **kwargs):
        """
        Args:
            maxlen (int):        the maximum sequence length to pad to
        """
        maxlen = kwargs.get('maxlen', None)
        if np.issubdtype(type(maxlen), np.integer):
            def func(x):
                ret = np.zeros([maxlen, 26], dtype=float)
                ret[np.arange(x.shape[0]), x] = 1.0
                return ret
        else:
            def func(x):
                ret = np.zeros([x.shape[0], 26], dtype=float)
                ret[np.arange(x.shape[0]), x] = 1.0
                return ret
        return func

    def get_torch_conversion(self, **kwargs):
        """
        Args:
            dtype:               the dtype to return data as
            device:              the device to send data to
            maxlen (int):        the maximum sequence length to pad to
        """
        dtype = kwargs.get('dtype')
        device = kwargs.get('device')
        maxlen = kwargs.get('maxlen')
        if kwargs.get('ohe', False):
            if np.issubdtype(type(maxlen), np.integer):
                def func(x):
                    ret = torch.zeros([maxlen, 26], dtype=dtype, device=device)
                    ret[np.arange(x.shape[0]), x.tolist()] = 1.0
                    ret = ret.T
                    return ret
            else:
                def func(x):
                    ret = torch.zeros([x.shape[0], 26], dtype=dtype, device=device)
                    ret[np.arange(x.shape[0]), x.tolist()] = 1.0
                    ret = ret.T
                    return ret
        else:
            if np.issubdtype(type(maxlen), np.integer):
                def func(x):
                    ret = torch.tensor(x, device=device, dtype=torch.int64)
                    return ret
            else:
                def func(x):
                    ret = torch.zeros(maxlen, device=device, dtype=torch.int64)
                    ret[0:x.shape[0]] = x
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
class TaxaTable(DynamicTable, TorchableMixin):

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'taxon_id', 'type': ('array_data', 'data', VectorData), 'doc': 'the taxon ID'},
            {'name': 'embedding', 'type': ('array_data', 'data', VectorData), 'doc': 'the embedding for each taxon'},
            {'name': 'phylum', 'type': ('array_data', 'data', VectorData), 'doc': 'the phylum for each taxon'},
            {'name': 'class', 'type': ('array_data', 'data', VectorData), 'doc': 'the class for each taxon'},
            {'name': 'order', 'type': ('array_data', 'data', VectorData), 'doc': 'the order for each taxon'},
            {'name': 'family', 'type': ('array_data', 'data', VectorData), 'doc': 'the family for each taxon'},
            {'name': 'genus', 'type': ('array_data', 'data', VectorData), 'doc': 'the genus for each taxon'},
            {'name': 'species', 'type': ('array_data', 'data', VectorData), 'doc': 'the species for each taxon'})
    def __init__(self, **kwargs):
        taxon_id, embedding = popargs('taxon_id', 'embedding', kwargs)
        taxonomy_labels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        taxonomy = popargs(*taxonomy_labels, kwargs)

        columns = list()
        if isinstance(taxon_id, VectorData):      # data is being read -- here we assume that everything is a VectorData
            columns.append(taxon_id)
            columns.append(embedding)
            columns.extend(taxonomy)
        else:
            columns.append(VectorData('taxon_id', 'taxonomy IDs from NCBI', data=taxon_id))
            columns.append(VectorData('embedding', 'an embedding for each taxon', data=embedding))
            for l, t in zip(taxonomy_labels, taxonomy):
                columns.append(VectorData(l, 'the %s for each taxon' % l, data=t))
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
        self._sanity = False

    def set_sanity(self, sanity):
        self._sanity = sanity

    def __getitem__(self, i):
        """
        Return a tuple containing (taxon_name, sequence_name, sequence, taxon_embedding)
        """
        (seq_i, seq_name, sequence, length,
         (tax_i, taxon_name, taxon_emb, p, c, o, f, g, s)) = self.seq_table[i]
        #breakpoint()
        if self._sanity:
            s = [0 for i in range(len(sequence.shape))]
            s[-1] = np.s_[0:len(taxon_emb)]
            sequence[s] = taxon_emb
            sequence = sequence[0:5, 0:100]

        return {'taxon': taxon_name, 'name': seq_name, "sequence": sequence, "embedding": taxon_emb}

    def __len__(self):
        return len(self.seq_table)

    def set_torch(self, use_torch, dtype=None, device=None, ohe=True, pad=False):
        maxlen = None
        if pad:
            maxlen = np.max(self.seq_table['length'][:])
        self.seq_table.set_torch(use_torch, dtype=dtype, device=device, ohe=ohe, maxlen=maxlen)
        self.taxa_table.set_torch(use_torch, dtype=dtype, device=device)

    def set_raw(self):
        self.seq_table.set_raw()

    def to_sequence(self, data):
        return self.seq_table.to_sequence(data)

    @staticmethod
    def _to_numpy(data):
        return data[:]

    @staticmethod
    def _to_torch(device=None, dtype=None):
        def func(data):
            return torch.tensor(data, device=device, dtype=dtype)
        return func

    def load(self, torch=False, device=None):
        for c in self.seq_table.columns:
            c.transform(self._to_numpy)
        for c in self.taxa_table.columns:
            c.transform(self._to_numpy)
        if torch:
            self.seq_table['sequence'].target.transform(self._to_torch(device))
            self.taxa_table['embedding'].transform(self._to_torch(device))
