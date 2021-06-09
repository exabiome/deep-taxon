from abc import ABCMeta, abstractmethod
import math

import numpy as np
import torch
import torch.nn.functional as F
import sklearn.neighbors as skn

from hdmf.common import VectorIndex, VectorData, DynamicTable, CSRMatrix,\
                        DynamicTableRegion, register_class, EnumData
from hdmf.utils import docval, call_docval_func, get_docval, popargs
from hdmf.data_utils import DataIO
from hdmf import Container, Data


__all__ = ['DeepIndexFile',
           'AbstractChunkedDIFile',
           'WindowChunkedDIFile',
           'RevCompFilter',
           'SequenceTable',
           'GenomeTable',
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
            {'name': 'sequence', 'type': ('array_data', 'data', EnumData), 'doc': 'bitpacked DNA sequence'},
            {'name': 'sequence_index', 'type': ('array_data', 'data', VectorIndex), 'doc': 'index for sequence'},
            {'name': 'length', 'type': ('array_data', 'data', VectorData), 'doc': 'lengths of sequence'},
            {'name': 'genome', 'type': ('array_data', 'data', VectorData), 'doc': 'the genome for each table'},
            {'name': 'genome_table', 'type': DynamicTable, 'doc': "target for 'genome'", 'default': None},
            {'name': 'pad', 'type': bool, 'doc': 'pad sequences to the maximum length sequence', 'default': False},
            {'name': 'bitpacked', 'type': bool, 'doc': 'sequence data are packed', 'default': True},
            {'name': 'vocab', 'type': 'array_data', 'doc': 'the characters in the sequence vocabulary.', 'default': None})
    def __init__(self, **kwargs):
        sequence_name, index, sequence, genome, genome_table = popargs('sequence_name',
                                                             'sequence_index',
                                                             'sequence',
                                                             'genome',
                                                             'genome_table',
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
            columns.append(genome)
            if self.pad:   # if we need to pad, compute the maxlen
                self.maxlen = np.max(seqlens.data[:])
        else:
            columns.append(VectorData('sequence_name', 'sequence names', data=sequence_name))
            seq = self.get_sequence_data(sequence)
            columns.append(self.get_sequence_index(index, seq))
            columns.append(seq)
            columns.append(seq.elements)
            columns.append(VectorData('length', 'sequence lengths', data=seqlens))
            columns.append(DynamicTableRegion('genome', genome, 'the genome of each sequence', genome_table))
        kwargs['columns'] = columns
        call_docval_func(super().__init__, kwargs)


@register_class('SequenceTable', NS)
class SequenceTable(AbstractSequenceTable):

    def get_sequence_index(self, index, data):
        return VectorIndex('sequence_index', index, data)

    def get_sequence_data(self, data):
        if isinstance(data, DataIO):
            vocab = data.data.data.encoded_vocab
        else:
            vocab = self.vocab
        return EnumData('sequence', 'sequence data from a vocabulary', data=data, elements=vocab)

    dna = ['A', 'C', 'G', 'T', 'N']

    protein = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'J', 'K', 'L', 'M', 'N',
               'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'sequence_name', 'type': ('array_data', 'data', VectorData), 'doc': 'sequence names'},
            {'name': 'sequence', 'type': ('array_data', 'data', EnumData), 'doc': 'bitpacked DNA sequence'},
            {'name': 'sequence_index', 'type': ('array_data', 'data', VectorIndex), 'doc': 'index for sequence'},
            {'name': 'length', 'type': ('array_data', 'data', VectorData), 'doc': 'lengths of sequence'},
            {'name': 'genome', 'type': ('array_data', 'data', VectorData), 'doc': 'the genome of each sequence'},
            {'name': 'genome_table', 'type': DynamicTable, 'doc': "target for 'genomes'", 'default': None},
            {'name': 'pad', 'type': bool, 'doc': 'pad sequences to the maximum length sequence', 'default': False},
            {'name': 'vocab', 'type': ('array_data', str), 'doc': 'the characters in the sequence vocabulary. '\
                                                                  '*dna* for nucleic acids, *protein* for default amino acids',
             'default': 'dna'}, )
    def __init__(self, **kwargs):
        vocab = popargs('vocab', kwargs)
        self.vocab_type = vocab
        if vocab is not None:
            if isinstance(vocab, str):
                if vocab == 'dna':
                    vocab = self.dna
                elif vocab == 'protein':
                    vocab = self.protein
        self.vocab = vocab
        super().__init__(**kwargs)


@register_class('DNAData', NS)
class DNAData(EnumData):

    def get(self, key, **kwargs):
        return super().get(key, **kwargs)


@register_class('DNATable', NS)
class DNATable(SequenceTable):

    def get_sequence_data(self, data):
        vocab = self.vocab
        return DNAData('sequence', 'sequence data from a vocabulary', data=data, elements=vocab)

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
        {'name': 'phylum', 'description': 'the phylum for each taxon', 'enum': True},
        {'name': 'class', 'description': 'the class for each taxon', 'enum': True},
        {'name': 'order', 'description': 'the order for each taxon', 'enum': True},
        {'name': 'family', 'description': 'the family for each taxon', 'enum': True},
        {'name': 'genus', 'description': 'the genus for each taxon', 'enum': True},
        {'name': 'species', 'description': 'the species for each taxon'},
        {'name': 'embedding', 'description': 'the embedding for each taxon'},
        {'name': 'rep_taxon_id', 'description': 'the taxon ID for the this species representative'}
    )

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'taxon_id', 'type': ('array_data', 'data', VectorData), 'doc': 'the taxon ID'},
            {'name': 'phylum', 'type': ('array_data', 'data', EnumData), 'doc': 'the phylum for each taxon'},
            {'name': 'class', 'type': ('array_data', 'data', EnumData), 'doc': 'the class for each taxon'},
            {'name': 'order', 'type': ('array_data', 'data', EnumData), 'doc': 'the order for each taxon'},
            {'name': 'family', 'type': ('array_data', 'data', EnumData), 'doc': 'the family for each taxon'},
            {'name': 'genus', 'type': ('array_data', 'data', EnumData), 'doc': 'the genus for each taxon'},
            {'name': 'species', 'type': ('array_data', 'data', VectorData), 'doc': 'the species for each taxon'},
            {'name': 'embedding', 'type': ('array_data', 'data', VectorData), 'doc': 'the embedding for each taxon', 'default': None},
            {'name': 'rep_taxon_id', 'type': ('array_data', 'data', VectorData), 'doc': 'the taxon ID for the species representative', 'default': None})
    def __init__(self, **kwargs):
        taxon_id, embedding, rep_taxon_id = popargs('taxon_id', 'embedding', 'rep_taxon_id', kwargs)
        taxonomy_labels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        taxonomy = popargs(*taxonomy_labels, kwargs)
        self.__taxmap = {t: i for i, t in enumerate(taxonomy_labels)}

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
                if isinstance(t, EnumData):
                    columns.append(t)
                    columns.append(t.elements)
                elif isinstance(t, VectorData):
                    columns.append(t)
                else:
                    columns.append(VectorData(l, 'the %s for each taxon' % l, data=t))
        kwargs['columns'] = columns
        call_docval_func(super().__init__, kwargs)

    def get_outputs_map(self, in_tax, out_tax):
        """
        Return a mapping from the *out_tax* taxonomic level to the *in_tax* taxonomic level

        See exabiome.nn.model.resnet.ResNet.reconfigure_outputs for an example of where this gets used
        Args:
            in_tax (str)        : the input taxonomic level
            out_tax (str)       : the output taxonomic level
        """
        if in_tax not in self.__taxmap:
            raise ValueError(f'Unrecognized taxonomic level: {in_tax}')
        if out_tax not in self.__taxmap:
            raise ValueError(f'Unrecognized taxonomic level: {out_tax}')
        if self.__taxmap[in_tax] >= self.__taxmap[out_tax]:
            raise ValueError(f'got in_tax={in_tax} and out_tax={out_tax} -- in_tax should be a higher taxonomic level than out_tax')
        in_ids = np.asarray(self[in_tax].data)
        out_ids = np.asarray(self[out_tax].data)
        ret = np.ones(len(np.unique(out_ids)), dtype=int) * -1
        for in_id in np.unique(in_ids):
            mask = in_ids == in_id
            out_mask = np.unique(out_ids[mask])
            if any(ret[out_mask] != -1):
                raise ValueError("hierarchy violation!")
            ret[out_mask] = in_id
        return ret

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
            ret = super().__getitem__(key)
            return ret


@register_class('GenomeTable', NS)
class GenomeTable(DynamicTable):
    __columns__ = (
        {'name': 'taxon_id', 'description': 'the taxon ID'},
        {'name': 'rep_idx', 'description': 'the index for the representative taxon', 'table': True}
    )

    @docval(*get_docval(DynamicTable.__init__),
            {'name': 'taxon_id', 'type': ('array_data', 'data', VectorData), 'doc': 'the taxon ID'},
            {'name': 'rep_idx', 'type': ('array_data', 'data', VectorData),
             'doc': 'the taxon ID for the species representative'},
            {'name': 'taxa_table', 'type': TaxaTable, 'doc': 'the table storing taxa information', 'default': None})
    def __init__(self, **kwargs):
        columns = kwargs['columns']
        if columns is None:
            taxon_id, rep_idx, taxa_table = popargs('taxon_id', 'rep_idx', 'taxa_table', kwargs)
            if not isinstance(taxon_id, VectorData):
                taxon_id = VectorData('taxon_id', 'NCBI accession', data=taxon_id)
            if not isinstance(rep_idx, VectorData):
                if taxa_table is None:
                    raise ValueError('taxa_table must be provided if writing')
                rep_idx = DynamicTableRegion('rep_idx', rep_idx, 'representative taxon', taxa_table)
            kwargs['columns'] = [taxon_id, rep_idx]
        call_docval_func(super().__init__, kwargs)


@register_class('CondensedDistanceMatrix', NS)
class CondensedDistanceMatrix(Data):
    pass


@register_class('NewickString', NS)
class NewickString(Data):
    pass


@register_class('TreeGraph', NS)
class TreeGraph(CSRMatrix):

    __fields__ = ('leaves', 'table')

    @docval(*get_docval(CSRMatrix.__init__),
            {'name': 'leaves', 'type': ('array_data', 'data'), 'doc': ('the index into *table* for each node in the graph. '
                                                                       'Internodes (i.e. non-leaf nodes) should have a -1')},
            {'name': 'table', 'type': GenomeTable, 'doc': 'the GenomeTable that the leaves in the tree belong to'})
    def __init__(self, **kwargs):
        leaves, table = popargs('leaves', 'table', kwargs)
        call_docval_func(super().__init__, kwargs)
        self.leaves = leaves
        self.table = table


@register_class('DeepIndexFile', NS)
class DeepIndexFile(Container):

    taxonomic_levels = ("phylum", "class", "order", "family", "genus", "species")

    __fields__ = ({'name': 'seq_table', 'child': True},
                  {'name': 'taxa_table', 'child': True},
                  {'name': 'genome_table', 'child': True},
                  {'name': 'tree_graph', 'child': True},
                  {'name': 'distances', 'child': True},
                  {'name': 'tree', 'child': True})

    @docval({'name': 'seq_table', 'type': (AATable, DNATable, SequenceTable), 'doc': 'the table storing DNA sequences'},
            {'name': 'taxa_table', 'type': TaxaTable, 'doc': 'the table storing taxa information'},
            {'name': 'genome_table', 'type': GenomeTable, 'doc': 'the table storing taxonomic information about species in this file'},
            {'name': 'tree', 'type': NewickString, 'doc': 'the table storing taxa information'},
            {'name': 'tree_graph', 'type': TreeGraph, 'doc': 'the graph representation of the tree'},
            {'name': 'distances', 'type': CondensedDistanceMatrix, 'doc': 'the table storing taxa information', 'default': None})
    def __init__(self, **kwargs):
        seq_table, taxa_table, genome_table, distances, tree, tree_graph = popargs('seq_table', 'taxa_table', 'genome_table',
                                                                                   'distances', 'tree', 'tree_graph', kwargs)
        call_docval_func(super().__init__, {'name': 'root'})
        self.seq_table = seq_table
        self.taxa_table = taxa_table
        self.genome_table = genome_table
        self.tree_graph = tree_graph
        self.__n_outputs = len(taxa_table)
        self.distances = distances
        self.tree = tree
        self._sanity = False
        self._sanity_features = 5
        self.__labels = None
        self.__n_emb_components = self.taxa_table['embedding'].data.shape[1] if 'embedding' in self.taxa_table else 0
        self.set_label_key('id')
        self.__rev = False
        self.__get_kwargs = dict()

    def set_label_key(self, val):
        self.label_key = val
        if val == 'species':         # if species is specified, just use the id column
            self.label_key = 'id'
        elif val in self.taxonomic_levels:
            self.__get_kwargs['index'] = True
            self.__n_outputs = len(self.taxa_table[self.label_key].elements)

    @property
    def n_outputs(self):
        return self.__n_outputs

    def set_sanity(self, sanity, n_features=5):
        self._sanity = sanity
        self._sanity_features = n_features

    def set_revcomp(self, revcomp=True):
        if revcomp and self.seq_table.vocab_type != 'dna':
                raise ValueError("Can only set reverse complement on DNA sequence data")
        self.__rev = revcomp

    @property
    def labels(self):
        if self.__labels is None:
            reps = self.genome_table['rep_idx'].data[:]
            genome_idx = self.seq_table['genome'].data[:]
            self.__labels = reps[genome_idx]
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
        label = self.seq_table['genome'].get(arg, index=True)    # index to genome table
        label = self.genome_table['rep_idx'].get(label, index=True) # index to taxa table
        label = self.taxa_table[self.label_key].get(label, **self.__get_kwargs)
        length = self.seq_table['length'].get(arg)
        return {'id': idx, 'seq': seq, 'label': label, 'length': length}

    def __len__(self):
        return len(self.seq_table) * (2 if self.__rev else 1)

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

class DIFileFilter(object):

    def __init__(self, difile):
        self.difile = difile

    def __getattr__(self, attr):
        """Delegate retrival of attributes to the data in self.data"""
        return getattr(self.difile, attr)

class AbstractChunkedDIFile(DIFileFilter):
    """
    An abstract class for chunking sequences from a DeepIndexFile
    """

    def __init__(self, difile, seq_idx, start, end, labels):
        super().__init__(difile)
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
        item['seq'] = item['seq'][s:e]
        item['seq_idx'] = seq_i
        item['id'] = i
        # item['seq_name'] += f'|{s}-{e}'
        item['length'] = e - s
        return item


class WindowChunkedDIFile(AbstractChunkedDIFile):
    """
    A class for chunking sequences with a sliding window

    By default windows are not overlapping
    """

    def __init__(self, difile, wlen, step=None, min_seq_len=100):
        if not isinstance(difile, (DeepIndexFile, DIFileFilter)):
            raise ValueError(f'difile must be a DeepIndexFile or a DIFileFilter, got {type(difile)}')
        if step is None:
            step = wlen
        self.wlen = wlen
        self.step = step
        self.min_seq_len = min_seq_len

        lengths = difile.seq_table['length'][:].astype(int)
        # compute the number of chunks proced by each sequecne by adding
        # the number of full chunks in each sequence to the number of incomplete chunks
        n_chunks = ((lengths // self.step) +
                    (lengths % self.step > 0))                  # the number of chunks each sequence will produce
        labels = np.repeat(difile.labels, n_chunks)             # the labels for each chunks
        seq_idx = np.repeat(np.arange(len(n_chunks)), n_chunks) # the index of the sequence for each chunk
        chunk_start = list()
        for i in range(len(difile)):
            chunk_start.append(np.arange(0, lengths[i], self.step))
        chunk_start = np.concatenate(chunk_start)               # the start of each chunk in it's respective sequence
        chunk_end = chunk_start + self.wlen                     # the end of each chunk in it's respective sequence
        chunk_end = np.min(np.array([chunk_end,                 # trim any ends that go over the end of a sequence
                                     np.repeat(lengths, n_chunks)]), axis=0)

        mask = (chunk_end - chunk_start) >= self.min_seq_len    # get rid of any sequences that are less than the minimum length
        labels = labels[mask]
        seq_idx = seq_idx[mask]
        chunk_start = chunk_start[mask]
        chunk_end = chunk_end[mask]

        super().__init__(difile, seq_idx, chunk_start, chunk_end, labels)


class RevCompFilter(DIFileFilter):

    rcmap = torch.tensor([ 9, 10, 11, 12, 13, 14, 15, 16, 17,
                           0,  1,  2,  3,  4,  5,  6,  7,  8])

    chars = {
        'A': 'T',
        'G': 'C',
        'C': 'G',
        'T': 'A',
        'Y': 'R',
        'R': 'Y',
        'W': 'W',
        'S': 'S',
        'K': 'M',
        'M': 'K',
        'D': 'H',
        'V': 'B',
        'H': 'D',
        'B': 'V',
        'X': 'X',
        'N': 'N',
    }

    @classmethod
    def get_revcomp_map(cls, vocab):
        d = {c: i for i, c in enumerate(vocab)}
        rcmap = np.zeros(len(vocab), dtype=int)
        for i, base in enumerate(vocab):
            rc_base = cls.chars[base]
            base_i = d[base]
            rc_base_i = d[rc_base]
            rcmap[base_i] = rc_base_i
        return rcmap

    def __init__(self, difile):
        super().__init__(difile)
        self.labels = np.concatenate([self.labels, self.labels])
        vocab = difile.seq_table.sequence.elements.data
        self.rcmap = torch.as_tensor(self.get_revcomp_map(vocab), dtype=torch.long)

    def __len__(self):
        return 2*len(self.difile)

    def __getitem__(self, arg):
        oarg = arg
        arg, rev = divmod(arg, 2)
        item = self.difile[arg]
        try:
            if rev:
                item['seq'] = self.rcmap[item['seq'].long()]
        except AttributeError as e:
            raise ValueError("Cannot run without loading data. Use -l to load data") from e
        item['id'] = oarg
        return item
