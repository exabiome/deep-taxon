from abc import ABCMeta, abstractmethod
import copy
from datetime import datetime
import math
import sys
import warnings

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sklearn.neighbors as skn

from hdmf.common import VectorIndex, VectorData, DynamicTable, CSRMatrix,\
                        DynamicTableRegion, register_class, EnumData
from hdmf.utils import docval, call_docval_func, get_docval, popargs
from hdmf.data_utils import DataIO
from hdmf import Container, Data
from hdmf.common import get_hdf5io

from ..utils import balsplit


__all__ = ['DeepIndexFile',
           'LazyWindowChunkedDIFile',
           'SequenceTable',
           'GenomeTable',
           'TaxaTable',
           'lazy_chunk_sequence',
           'chunk_sequence']

NS = 'deep-index'


def log(msg, print_msg=True):
    if print_msg:
        print(f'{datetime.now()} - {msg}', file=sys.stderr)


class AbstractSequenceTable(DynamicTable, metaclass=ABCMeta):

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
class TaxaTable(DynamicTable):

    __columns__ = (
        {'name': 'taxon_id', 'description': 'the taxon ID'},
        {'name': 'domain', 'description': 'the domain of each taxon', 'enum': True},
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
            {'name': 'domain', 'type': ('array_data', 'data', EnumData), 'doc': 'the domain for each taxon'},
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
        taxonomy_labels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
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

        See deep_taxon.nn.model.resnet.ResNet.reconfigure_outputs for an example of where this gets used
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
        if out_tax == 'species':
            out_ids = np.arange(len(self))
        else:
            out_ids = np.asarray(self[out_tax].data)
        ret = np.ones(len(np.unique(out_ids)), dtype=int) * -1
        for in_id in np.unique(in_ids):
            mask = in_ids == in_id
            out_mask = np.unique(out_ids[mask])
            if any(ret[out_mask] != -1):
                raise ValueError("hierarchy violation!")
            ret[out_mask] = in_id
        return ret

    def get_num_classes(self, tax_lvl):
        if tax_lvl == 'species':
            return len(self)
        else:
            return len(self[tax_lvl].elements)

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

    taxonomic_levels = ("domain", "phylum", "class", "order", "family", "genus", "species")

    __fields__ = ({'name': 'seq_table', 'child': True},
                  {'name': 'taxa_table', 'child': True},
                  {'name': 'genome_table', 'child': True},
                  {'name': 'tree_graph', 'child': True},
                  {'name': 'distances', 'child': True},
                  {'name': 'tree', 'child': True})

    @docval({'name': 'seq_table', 'type': (AATable, DNATable, SequenceTable), 'doc': 'the table storing DNA sequences'},
            {'name': 'taxa_table', 'type': TaxaTable, 'doc': 'the table storing taxa information'},
            {'name': 'genome_table', 'type': GenomeTable, 'doc': 'the table storing taxonomic information about species in this file'},
            {'name': 'tree', 'type': NewickString, 'doc': 'the table storing taxa information', 'default': None},
            {'name': 'tree_graph', 'type': TreeGraph, 'doc': 'the graph representation of the tree', 'default': None},
            {'name': 'distances', 'type': CondensedDistanceMatrix, 'doc': 'the table storing taxa information', 'default': None})
    def __init__(self, **kwargs):
        seq_table, taxa_table, genome_table, distances, tree, tree_graph = popargs('seq_table', 'taxa_table', 'genome_table',
                                                                                   'distances', 'tree', 'tree_graph', kwargs)
        call_docval_func(super().__init__, {'name': 'root'})
        self.seq_table = seq_table
        self.taxa_table = taxa_table
        self.genome_table = genome_table
        self.tree_graph = tree_graph
        self.distances = distances
        self.tree = tree
        self._sanity = False
        self._sanity_features = 5
        self._labels = None
        self.__loaded = False
        self.__indices = None
        self.__n_outputs = None
        self.__n_emb_components = self.taxa_table['embedding'].data.shape[1] if 'embedding' in self.taxa_table else 0
        self.set_label_key('id')
        self.__rev = False
        self.__get_kwargs = dict()
        self._vocab = np.chararray.upper(self.seq_table['sequence'].target.elements[:].astype('U1'))

    def set_label_key(self, val):
        self.label_key = val
        if val in ('species', 'id'):         # if species is specified, just use the id column
            self.label_key = 'id'
            genome_labels = self.genome_table['rep_idx'].data[:]
            self.__n_outputs = len(self.taxa_table)
        elif val in self.taxonomic_levels:
            self.__get_kwargs['index'] = True
            self.__n_outputs = len(self.taxa_table[self.label_key].elements)
            genome_labels = self.genome_table['rep_idx'].data[:]
            genome_labels = self.taxa_table[val].data[:][genome_labels]
        elif val == 'all':                   # use all taxonomic levels as labels
            cols = list()
            genome_labels = self.genome_table['rep_idx'].data[:]
            for lvl in self.taxonomic_levels[:-1]:
                cols.append(self.taxa_table[lvl].data[:][genome_labels])
            cols.append(self.taxa_table['id'].data[:][genome_labels])
            genome_labels = np.column_stack(cols)
        else:
            raise ValueError("Unrecognized label key: '%s'" % val)
        if self.__indices is None:
            genome_idx = self.seq_table['genome'].data[:]
        else:
            genome_idx = self.seq_table['genome'].data[self.__indices]
        self._labels = genome_labels[genome_idx]

        if self.label_key in ('species', 'id', 'all'):
            self._classes = self.taxa_table['species'].data[:]
        else:
            self._classes = self.taxa_table[self.label_key].elements.data[:]

    def get_label_classes(self):
        return self._classes

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
    def n_emb_components(self):
        return self.__n_emb_components

    def __getitem__(self, i):
        """
        Return a tuple containing (taxon_name, sequence_name, sequence, taxon_embedding)
        """
        return self.get(i)

    def get(self, arg):
        arg = self.__translate_arg(arg)
        idx = self.seq_table.id[arg]
        label = self._labels[arg]
        seq = self.seq_table['sequence'].get(arg, index=True)   # sequence data
        length = self.seq_table['length'].get(arg)
        return {'id': idx, 'seq': seq, 'label': label, 'length': length}

    def set_sequence_subset(self, indices=None):
        self.__indices = indices
        self._labels = self._labels[self.__indices]

    def load(self, sequence=False, device=None, verbose=True):
        indices = self.__indices
        self.__loaded = True
        if self.__indices is not None:

            log('Loading data subset - getting start/end for sequences in subset', print_msg=verbose)
            # get start/end of subset sequences
            dset = self.seq_table['sequence_index'].data.astype(int)[:]
            starts = dset[self.__indices - 1]
            if self.__indices[0] == 0:
                starts[0] = 0
            ends = dset[self.__indices]

            log('Loading data subset - computing lengths for sequences in subset', print_msg=verbose)
            # get lengths of the subset sequences
            subset_lengths = ends - starts

            log('Loading data subset - computing index for sequence subset', print_msg=verbose)
            # compute the index for the subset
            subset_index = np.cumsum(subset_lengths)

            # read each sequence in
            sequence_subset = np.zeros(subset_index[-1], dtype=np.uint8)
            s_dest, e_dest = 0, 0
            seq_dset = self.seq_table['sequence'].target.data
            ## read one sequence at a time
            it = zip(starts, ends)
            log('Loading data subset - loading sequences from subset', print_msg=verbose)
            if verbose:
                it = tqdm(it, total=len(starts))
            for s_src, e_src in it:
                e_dest = e_src - s_src + s_dest
                seq_dset.read_direct(sequence_subset, np.s_[s_src:e_src], np.s_[s_dest:e_dest])
                s_dest = e_dest

            # swap everything in
            log('Loading data subset - loading IDs', print_msg=verbose)
            self.seq_table['id'].transform(lambda x: x[:][self.__indices])
            log('Loading data subset - swapping in lengths', print_msg=verbose)
            self.seq_table['length'].transform(lambda x: subset_lengths)
            log('Loading data subset - swapping in sequence index', print_msg=verbose)
            self.seq_table['sequence_index'].transform(lambda x: subset_index)
            if sequence:
                log('Loading data subset - swapping in sequence data', print_msg=verbose)
                self.seq_table['sequence_index'].target.transform(lambda x: sequence_subset)
            log('Loading data subset - done loading data subset', print_msg=verbose)
        else:
            _load = lambda x: x[:]
            log('Loading data - loading IDs', print_msg=verbose)
            self.seq_table['id'].transform(_load)
            log('Loading data - loading sequence lengths', print_msg=verbose)
            self.seq_table['length'].transform(lambda x: x.astype(int)[:])
            log('Loading data - loading sequence index', print_msg=verbose)
            self.seq_table['sequence_index'].transform(_load)
            if sequence:
                log('Loading data - loading sequences', print_msg=verbose)
                self.seq_table['sequence_index'].target.transform(_load)
            log('Loading data - done loading data', print_msg=verbose)

    def get_sequence_subset(self):
        return copy.copy(self.__indices)

    def __translate_arg(self, arg):
        """
        Translate from a subset index to the original sequence index
        """
        if self.__indices is None or self.__loaded:
            return arg
        else:
            return self.__indices[arg]

    def __len__(self):
        if self.__indices is None:
            return len(self.seq_table) * (2 if self.__rev else 1)
        else:
            return len(self.__indices)

    def get_seq_lengths(self):
        if self.__indices is None:
            if isinstance(self.seq_table['length'].data, np.ndarray):
                return self.seq_table['length'].data.copy()
            else: # h5py.Dataset
                return self.seq_table['length'].data[:]
        else:
            return self.seq_table['length'].data[self.__indices]

    def set_raw(self):
        self.seq_table.set_raw()

    def to_sequence(self, data):
        return self.seq_table.to_sequence(data)

    def set_subset(self, subset_counts, seed, starts=None):
        warnings.warn("Cannot subset DeepIndexFile -- ignoring")

    def get_vocab(self):
        return self._vocab

class DIFileFilter(object):

    def __init__(self, difile):
        self.difile = difile

    def __getattr__(self, attr):
        """Delegate retrival of attributes to the data in self.data"""
        return getattr(self.difile, attr)


def chunk_sequence(difile, wlen, step=None, min_seq_len=100):
    """
    Compute start and ends for a chunked sequence

    Args:
        difile (DeepIndexFile)      : the DeepIndexFile with the sequence data to chunk
        wlen (int)                  : the window length to chunk into
        step (int)                  : the step between window starts. by default this is wlen
                                      i.e. non-overlapping windows
        min_seq_len (int)           : the minimum sequence length to keep

    Returns:
        A tuple of (seq_idx, chunk-start, chunk_end, labels)
        seq_idx (array)             : the index of the sequence that each chunk is derived from
        chunk_start (array)         : the start of each chunk in its respective sequence
        chunk_end (array)           : the end of each chunk in its respective sequence
        labels (array)              : the taxonomic labels of each chunk
    """
    if not isinstance(difile, (DeepIndexFile, DIFileFilter)):
        raise ValueError(f'difile must be a DeepIndexFile or a DIFileFilter, got {type(difile)}')
    if step is None:
        step = wlen

    dtype = np.uint32

    lengths = difile.get_seq_lengths() # difile.seq_table['length'][:].astype(dtype)
    # compute the number of chunks proced by each sequecne by adding
    # the number of full chunks in each sequence to the number of incomplete chunks
    n_chunks = ((lengths // step) +
                (lengths % step > 0))                  # the number of chunks each sequence will produce
    labels = np.repeat(difile.labels.astype(dtype), n_chunks, axis=0)             # the labels for each chunks
    seq_idx = np.repeat(np.arange(len(n_chunks), dtype=dtype), n_chunks) # the index of the sequence for each chunk
    chunk_start = list()
    for i in range(len(difile)):
        chunk_start.append(np.arange(0, lengths[i], step, dtype=dtype))
    chunk_start = np.concatenate(chunk_start)               # the start of each chunk in it's respective sequence
    chunk_end = chunk_start + wlen                     # the end of each chunk in it's respective sequence
    chunk_end = np.min(np.array([chunk_end,                 # trim any ends that go over the end of a sequence
                                 np.repeat(lengths, n_chunks)]), axis=0)

    mask = (chunk_end - chunk_start) >= min_seq_len    # get rid of any sequences that are less than the minimum length
    labels = labels[mask]
    seq_idx = seq_idx[mask]
    chunk_start = chunk_start[mask]
    chunk_end = chunk_end[mask]
    return seq_idx, chunk_start, chunk_end, labels, mask.mean()


def lazy_chunk_sequence(difile, wlen, step=None, min_seq_len=100):
    if wlen < min_seq_len:
        raise ValueError('window size (wlen) must be greater than or equal to minimum chunk length (min_seq_len)')

    # the length of each sequence
    lengths = np.asarray(difile.get_seq_lengths(), dtype=int)

    # C_min is the shortest chunk before filtering (for each sequence)
    C_min = lengths % step
    C_min[C_min == 0] = step

    # n_short_C is the number of chunks that are shorter than M (for each sequence)
    n_short_C = np.maximum(((min_seq_len - C_min - 1) // step) + 1, 0)

    # n_C is the number of chunks in each sequence before filtering
    n_C = (lengths - 1) // step + 1

    # ret is the number of valid chunks (i.e. chunks >= min_seq_len) for each sequence
    ret = n_C - n_short_C
    frac_good = ret.sum() / n_C.sum()
    return ret, frac_good


class LazyWindowChunkedDIFile:
    """
    An abstract class for chunking sequences from a DeepIndexFile
    """

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


    def __init__(self, difile, window, step, min_seq_len=100, rank=0, size=1, revcomp=False, distances=False, tree_graph=False):
        counts, frac_good = lazy_chunk_sequence(difile, window, step, min_seq_len)
        if size > 1:
            indices = balsplit(counts, size, rank)
            counts = counts[indices]
            difile.set_sequence_subset(indices)

        difile.load(sequence=True, verbose=rank==0)
        log('setting lengths', print_msg=rank==0)
        self.lengths = np.asarray(difile.seq_table['length'].data, dtype=int)
        log('setting ids', print_msg=rank==0)
        self.id = np.asarray(difile.seq_table['id'].data, dtype=int)
        log('setting labels', print_msg=rank==0)
        self.labels = np.asarray(difile._labels)
        log('setting seq_index', print_msg=rank==0)
        self.seq_index = np.asarray(difile.seq_table['sequence_index'].data, dtype=int)
        log('setting sequence', print_msg=rank==0)
        self.sequence = np.asarray(difile.seq_table['sequence_index'].target.data, dtype=np.uint8)
        log('done setting important data', print_msg=rank==0)

        self.n_outputs = difile.n_outputs
        self.classes = difile.get_label_classes()
        self.vocab = difile.get_vocab()

        self.distances = None
        if distances:
            self.distances = difile.distances.data[:]
        self.node_ids = None
        self.tree_graph = None
        if tree_graph:
            leaves = difile.tree_graph.leaves[:]
            self.node_ids = np.zeros(leaves.max()+1)
            for i in range(len(leaves)):
                tid = leaves[i]
                if i < 0:
                    continue
                self.node_ids[tid] = i
            self.tree_graph = difile.tree_graph.to_spmat()

        self.rcmap = None
        self.__rc = 1
        if revcomp:
            self.rcmap = torch.as_tensor(self.get_revcomp_map(self.vocab), dtype=torch.uint8)
            self.__rc = 2


        self.orig_lut = np.cumsum(counts)
        self.lut = self.orig_lut
        self.window = window
        self.step = step
        self.n_discarded = int(self.lut[-1] / frac_good - self.lut[-1])

        self.subset_counts = None
        self.seed = None
        self.starts = None
        log('done constructing LazyWindowDIFile', print_msg=rank==0)

    @property
    def n_seqs(self):
        return len(self.seq_index)

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

    def __get_helper(self, arg):
        if arg > len(self.seq_index):
            raise ValueError("seq_index {arg} out of bound for LazyWindowChunkedDIFile of length {len(self.seq_index)}")
        s = 0 if arg == 0 else self.seq_index[arg - 1]
        e = self.seq_index[arg]
        seq = self.sequence[s:e]

        idx = self.id[arg]
        label = self.labels[arg]
        length = self.lengths[arg]
        return {'id': idx, 'seq': seq, 'label': label, 'length': length}

    def get_label_classes(self):
        return self._classes

    def get_seq_lengths(self):
        self.lengths.copy()

    def get_counts(self, orig=False):
        """
        Return the chunk counts for each sequence

        Args:
            orig (bool)     : return counts for sequence regardless of whether not
                              this LazyWindowChunkedDIFile has been subsetting (with set_subset)
        """
        if orig:
            counts = self.orig_lut.copy()
        else:
            counts = self.lut.copy()
        counts[1:] = counts[1:] - counts[:-1]
        return counts

    def __len__(self):
        return self.lut[-1] * self.__rc

    def set_subset(self, subset_counts, seed, starts=None):
        """
        To reset, pass None for subset_counts
        """
        self.subset_counts = subset_counts
        if self.subset_counts is None:
            self.lut = self.orig_lut
            self.starts = None
            self.seed = None
        else:
            self.lut = np.cumsum(self.subset_counts)
            self.starts = starts
            self.seed = seed

    def __getitem__(self, i):
        if not isinstance(i, (int, np.integer)):
            raise ValueError("LazyWindowChunkedDIFile only supports indexing with an integer")

        # make sure i is a nonnegative integer
        i = i % len(self)
        rev = 0
        i, rev = divmod(i, self.__rc)

        idx = i
        if idx < 0:
            idx += self.lut[-1]
            if idx < 0:
                raise IndexError(f'index {i} is out of bounds for LazyWindowChunkedDIFile of length {self.lut[-1]}')

        seq_i = np.searchsorted(self.lut, idx, side="right")
        if seq_i == len(self.lut):
            raise IndexError(f'index {i} out of bounds for LazyWindowChunkedDIFile of length {self.lut[-1]}')
        chunk_i = idx if seq_i == 0 else idx - self.lut[seq_i - 1]


        # This section handles subsets (i.e. training vs validation). A subset is set by specifying the number of
        # chunks in each sequence that are part of the subset. This is stored in self.lut (see set_subset method).
        # The total number of chunks in a sequence is maintained in self.orig_lut. After the sequence to pull a chunk
        # from is determined using self.lut (see np.searchsorted call above), the chunk to retrieve is determined using
        # self.subset_counts and self.starts. self.subset_counts indicates how many chunks each sequence has in the subset,
        # and self.starts indicates where those chunks start. For example, for sequence i, the chunks in sequence i
        # that are in the subset are self.starts[i]:self.starts[i]+self.subset_counts[i].
        # To keep subset assignment random, the chunks are randomly permuted. We do this on the fly (see permutation
        # call below) to reduce storage overhead i.e. if we precomputed and stored them, the storage requirements for
        # chunking would grow linearly w.r.t. the number of chunks.
        if self.subset_counts is not None:
            if self.starts is not None:
                chunk_i += self.starts[seq_i]
            n_chunks = self.orig_lut[seq_i] if seq_i == 0 else self.orig_lut[seq_i] - self.orig_lut[seq_i-1]
            chunk_indices = np.random.default_rng(seed=self.seed + seq_i).permutation(n_chunks)
            chunk_i = chunk_indices[chunk_i]

        s = self.step * chunk_i
        l = self.lengths[seq_i]
        e = min(s + self.window, l)

        item = self.__get_helper(seq_i)
        item['seq_idx'] = item['id']

        if rev:
            item['seq'] = self.rcmap[item['seq'][l-e:l-s].astype(int)].flip(0)
        else:
            item['seq'] = item['seq'][s:e]
        item['id'] = i
        item['length'] = e - s
        return item
