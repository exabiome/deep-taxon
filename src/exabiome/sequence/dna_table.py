from abc import ABCMeta, abstractmethod
import math

import numpy as np
import torch
import torch.nn.functional as F
import sklearn.neighbors as skn

from hdmf.common import VectorIndex, VectorData, DynamicTable,\
                        DynamicTableRegion, register_class, VocabData
from hdmf.utils import docval, call_docval_func, get_docval, popargs
from hdmf.data_utils import DataIO
from hdmf import Container, Data


__all__ = ['DeepIndexFile',
           'AbstractChunkedDIFile',
           'WindowChunkedDIFile',
           'SequenceTable',
           'TaxaTable']

NS = 'deep-index'


class TorchableMixin:


    def __init__(self, *args, **kwargs):
        self.use_torch = False
        self.classify = False

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
                return torch.as_tensor(x, dtype=dtype, device=device)
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


class AbstractSequenceTable(DynamicTable, TorchableMixin, metaclass=ABCMeta):

    @abstractmethod
    def get_sequence_index(self, data, target):
        pass

    @abstractmethod
    def get_sequence_data(self, data):
        pass

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'sequence_name', 'type': ('array_data', 'data', VectorData), 'doc': 'sequence names'},
            {'name': 'sequence', 'type': ('array_data', 'data', VectorData), 'doc': 'bitpacked DNA sequence'},
            {'name': 'sequence_index', 'type': ('array_data', 'data', VectorIndex), 'doc': 'index for sequence'},
            {'name': 'length', 'type': ('array_data', 'data', VectorData), 'doc': 'lengths of sequence'},
            {'name': 'taxon', 'type': ('array_data', 'data', VectorData), 'doc': 'index for sequence'},
            {'name': 'taxon_table', 'type': DynamicTable, 'doc': "target for 'taxon'", 'default': None},
            {'name': 'pad', 'type': bool, 'doc': 'pad sequences to the maximum length sequence', 'default': False},
            {'name': 'bitpacked', 'type': bool, 'doc': 'sequence data are packed', 'default': True},
            {'name': 'vocab', 'type': 'array_data', 'doc': 'the characters in the sequence vocabulary.', 'default': None})
    def __init__(self, **kwargs):
        sequence_name, index, sequence, taxon, taxon_table = popargs('sequence_name',
                                                             'sequence_index',
                                                             'sequence',
                                                             'taxon',
                                                             'taxon_table',
                                                             kwargs)
        self._bitpacked = popargs('bitpacked', kwargs)
        vocab = popargs('vocab', kwargs)

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
            columns.append(self.get_sequence_data(sequence))
            columns.append(self.get_sequence_index(index, columns[-1]))
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
            return tuple(ret)


@register_class('SequenceTable', NS)
class SequenceTable(AbstractSequenceTable):

    def get_sequence_index(self, index, data):
        return VectorIndex('sequence_index', index, data)

    def get_sequence_data(self, data):
        if isinstance(data, DataIO):
            vocab = data.data.data.encoded_vocab
        else:
            vocab = self.vocab
        return VocabData('sequence', 'sequence data from a vocabulary', data=data, vocabulary=vocab)

    dna = ['A', 'C', 'G', 'T', 'N']

    protein = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'J', 'K', 'L', 'M', 'N',
               'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'sequence_name', 'type': ('array_data', 'data', VectorData), 'doc': 'sequence names'},
            {'name': 'sequence', 'type': ('array_data', 'data', VectorData), 'doc': 'bitpacked DNA sequence'},
            {'name': 'sequence_index', 'type': ('array_data', 'data', VectorIndex), 'doc': 'index for sequence'},
            {'name': 'length', 'type': ('array_data', 'data', VectorData), 'doc': 'lengths of sequence'},
            {'name': 'taxon', 'type': ('array_data', 'data', VectorData), 'doc': 'index for sequence'},
            {'name': 'taxon_table', 'type': DynamicTable, 'doc': "target for 'taxon'", 'default': None},
            {'name': 'pad', 'type': bool, 'doc': 'pad sequences to the maximum length sequence', 'default': False},
            {'name': 'vocab', 'type': ('array_data', str), 'doc': 'the characters in the sequence vocabulary. '\
                                                                  '*dna* for nucleic acids, *protein* for default amino acids',
             'default': 'dna'}, )
    def __init__(self, **kwargs):
        vocab = popargs('vocab', kwargs)
        if vocab is not None:
            if isinstance(vocab, str):
                if vocab == 'dna':
                    vocab = self.dna
                elif vocab == 'protein':
                    vocab = self.protein
        self.vocab = vocab
        super().__init__(**kwargs)


@register_class('DNAData', NS)
class DNAData(VocabData):

    def get(self, key, rev=False, **kwargs):
        idx = self.data[key]
        if rev:
            idx = (idx[::-1] + 9) % 18
        return self._get_helper(idx, **kwargs)


@register_class('DNATable', NS)
class DNATable(SequenceTable):

    def get_sequence_data(self, data):
        if isinstance(data, DataIO):
            vocab = data.data.data.encoded_vocab
        else:
            vocab = self.vocab
        return DNAData('sequence', 'sequence data from a vocabulary', data=data, vocabulary=vocab)

    def get_sequence_index(self, data, target):
        return VectorIndex('sequence_index', data, target)


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

    def get_sequence_index(self, data, target):
        return VectorIndex('sequence_index', data, target)

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

    __columns__ = (
        {'name': 'taxon_id', 'description': 'the taxon ID'},
        {'name': 'phylum', 'description': 'the phylum for each taxon'},
        {'name': 'class', 'description': 'the class for each taxon'},
        {'name': 'order', 'description': 'the order for each taxon'},
        {'name': 'family', 'description': 'the family for each taxon'},
        {'name': 'genus', 'description': 'the genus for each taxon'},
        {'name': 'species', 'description': 'the species for each taxon'},
        {'name': 'embedding', 'description': 'the embedding for each taxon'},
        {'name': 'rep_taxon_id', 'description': 'the taxon ID for the this species representative'}
    )

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'taxon_id', 'type': ('array_data', 'data', VectorData), 'doc': 'the taxon ID'},
            {'name': 'phylum', 'type': ('array_data', 'data', VectorData), 'doc': 'the phylum for each taxon'},
            {'name': 'class', 'type': ('array_data', 'data', VectorData), 'doc': 'the class for each taxon'},
            {'name': 'order', 'type': ('array_data', 'data', VectorData), 'doc': 'the order for each taxon'},
            {'name': 'family', 'type': ('array_data', 'data', VectorData), 'doc': 'the family for each taxon'},
            {'name': 'genus', 'type': ('array_data', 'data', VectorData), 'doc': 'the genus for each taxon'},
            {'name': 'species', 'type': ('array_data', 'data', VectorData), 'doc': 'the species for each taxon'},
            {'name': 'embedding', 'type': ('array_data', 'data', VectorData), 'doc': 'the embedding for each taxon', 'default': None},
            {'name': 'rep_taxon_id', 'type': ('array_data', 'data', VectorData), 'doc': 'the taxon ID for the species representative', 'default': None})
    def __init__(self, **kwargs):
        taxon_id, embedding, rep_taxon_id = popargs('taxon_id', 'embedding', 'rep_taxon_id', kwargs)
        taxonomy_labels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        taxonomy = popargs(*taxonomy_labels, kwargs)

        columns = list()
        if isinstance(taxon_id, VectorData):      # data is being read -- here we assume that everything is a VectorData
            columns.append(taxon_id)
            columns.extend(taxonomy)
            if embedding is not None: columns.append(embedding)
            if rep_taxon_id is not None: columns.append(rep_taxon_id)
        else:
            columns.append(VectorData('taxon_id', 'taxonomy IDs from NCBI', data=taxon_id))
            if embedding is not None: columns.append(VectorData('embedding', 'an embedding for each taxon', data=embedding))
            if rep_taxon_id is not None: columns.append(VectorData('rep_taxon_id', 'the taxon ID for the this species representative', data=rep_taxon_id))
            for l, t in zip(taxonomy_labels, taxonomy):
                columns.append(VectorData(l, 'the %s for each taxon' % l, data=t))
        kwargs['columns'] = columns
        call_docval_func(super().__init__, kwargs)
        self.convert = self.get_numpy_conversion()


    def taxid_torch_conversion(self, num_classes, device=None):
        def func(x):
            ret = torch.zeros(num_classes, dtype=torch.long, device=device)
            ret[x] = 1
            return ret
        return func

    def taxid_numpy_conversion(self, num_classes):
        def func(x):
            ret = np.zeros(num_classes, dtype=np.uint8)
            ret[x] = 1
            return ret
        return func

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        else:
            ret = list(super().__getitem__(key))
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

    @docval({'name': 'seq_table', 'type': (AATable, DNATable, SequenceTable), 'doc': 'the table storing DNA sequences'},
            {'name': 'taxa_table', 'type': TaxaTable, 'doc': 'the table storing taxa information'},
            {'name': 'tree', 'type': NewickString, 'doc': 'the table storing taxa information'},
            {'name': 'distances', 'type': CondensedDistanceMatrix, 'doc': 'the table storing taxa information', 'default': None})
    def __init__(self, **kwargs):
        seq_table, taxa_table, distances, tree = popargs('seq_table', 'taxa_table', 'distances', 'tree', kwargs)
        call_docval_func(super().__init__, {'name': 'root'})
        self.seq_table = seq_table
        self.taxa_table = taxa_table
        self.distances = distances
        self.tree = tree
        self._sanity = False
        self._sanity_features = 5
        self.__labels = None
        self.__n_emb_components = self.taxa_table['embedding'].data.shape[1] if 'embedding' in self.taxa_table else 0
        self.label_key = 'id'

    def set_label_key(self, val):
        self.label_key = val

    def set_sanity(self, sanity, n_features=5):
        self._sanity = sanity
        self._sanity_features = n_features

    @property
    def labels(self):
        if self.__labels is None:
            self.__labels = self.seq_table['taxon'].data[:]
        return self.__labels

    @property
    def n_emb_components(self):
        return self.__n_emb_components

    def __getitem__(self, i):
        """
        Return a tuple containing (taxon_name, sequence_name, sequence, taxon_embedding)
        """
        return self.get(i)

    def get(self, arg):
        idx = self.seq_table.id[arg]
        seq = self.seq_table['sequence'].get(arg, index=True)   # sequence data
        label = self.seq_table['taxon'].get(arg, index=True)    # index to taxon table
        label = self.taxa_table[self.label_key][label]          # get the interesting information from the taxon table i.e. embedding
        return (idx, seq, label)

    def __len__(self):
        return len(self.seq_table)

    def set_raw(self):
        self.seq_table.set_raw()

    def to_sequence(self, data):
        return self.seq_table.to_sequence(data)

    def get_knn_classifier(self, **kwargs):
        """
        Build a KNeighborsClassifier from taxonomic embeddings

        By default, use only a single neighbor

        Args:
            kwargs  : arguments to sklearn.neighbors.KNeighborsClassifier constructor


        """
        kwargs.setdefault('n_neighbors', 1)
        ret = skn.KNeighborsClassifier(**kwargs)
        emb = self.taxa_table['embedding'][:]
        ret.fit(emb, np.arange(emb.shape[0]))
        return ret



class AbstractChunkedDIFile(object):
    """
    An abstract class for chunking sequences from a DeepIndexFile
    """

    def __init__(self, difile, seq_idx, start, end, labels):
        self.difile = difile
        self.seq_idx = np.asarray(seq_idx)
        self.start = np.asarray(start)
        self.end = np.asarray(end)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.seq_idx)

    def __getitem__(self, i):
        if not isinstance(i, (int, np.integer)):
            raise ValueError("ChunkedDIFile only supportsd indexing with an integer")
        seq_i = self.seq_idx[i]
        item = self.difile[seq_i]
        s = self.start[i]
        e = self.end[i]
        #return seq_i, item[1][s:e], item[2]
        return i, item[1][s:e], item[2], seq_i

    def __getattr__(self, attr):
        """Delegate retrival of attributes to the data in self.data"""
        return getattr(self.difile, attr)


class WindowChunkedDIFile(AbstractChunkedDIFile):
    """
    A class for chunking sequences with a sliding window

    By default windows are not overlapping
    """

    def __init__(self, difile, wlen, step=None):
        if not isinstance(difile, DeepIndexFile):
            raise ValueError(f'difile must be a DeepIndexFile, got {type(difile)}')
        if step is None:
            step = wlen
        self.wlen = wlen
        self.step = step

        seq_idx = list()
        chunk_start = list()
        chunk_end = list()
        labels = list()

        lengths = difile.seq_table['length'][:]
        seqlabels = difile.labels
        for i in range(len(difile)):
            label = seqlabels[i]
            for start in range(0, lengths[i], step):
                labels.append(label)
                seq_idx.append(i)
                chunk_start.append(start)
                chunk_end.append(start+step)

        super().__init__(difile, seq_idx, chunk_start, chunk_end, labels)
