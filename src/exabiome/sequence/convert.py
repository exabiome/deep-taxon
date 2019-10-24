import h5py
import skbio.io
from skbio.sequence import DNA
import re
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import h5py
import os
import math
from collections import deque

class SeqIterator(object):
    ltag_re = re.compile('\[locus_tag=([A-Za-z0-9_]+)\]')

    def __init__(self, paths):
        self.dna_chars = 'ATCGN'

        chars = self.dna_chars
        chars = list(chars + chars.lower())
        self.nchars = len(chars)//2

        self.ohe = OneHotEncoder(sparse=False)
        self.ohe.fit(np.array(chars).reshape(-1, 1))
        categories = self.ohe.categories_[0][:self.nchars]

        self._col2drop = categories == 'N'
        self._row_mask = np.zeros(len(categories), dtype=bool)
        self._row_mask[np.logical_not(self._col2drop)] = True

        if isinstance(paths, str):
            paths = [paths]

        self.__paths = paths
        self.__path_iter = None
        self.__name_queue = None
        self.__index_queue = None
        self.__id_queue = None
        self.__total_len = 0
        self.__nseqs = 0

        self.__curr_block = np.zeros((0,4), dtype=np.uint8)
        self.__curr_block_idx = 0
        self.__padded = False

    @property
    def padded(self):
        return self.__padded

    @property
    def names(self):
        return self.__name_queue

    @property
    def index(self):
        return self.__index_queue

    @property
    def ids(self):
        return self.__id_queue

    def index_iter(self):
        return QueueIterator(self.index)

    def names_iter(self):
        return QueueIterator(self.names)

    def id_iter(self):
        return QueueIterator(self.ids)

    def __iter__(self):
        self.__path_iter = iter(self.__paths)
        # initialize the sequence iterator
        self.__curr_iter = self.__read(next(self.__path_iter))
        self.__name_queue = deque()
        self.__index_queue = deque()
        self.__id_queue = deque()
        self.__total_len = 0
        self.__nseqs = 0
        self.__curr_block = np.zeros((0,4), dtype=np.uint8)
        self.__curr_block_idx = 0
        return self

    def __read_next_seq(self):
        while True:
            try:
                seq, seqname = next(self.__curr_iter)
                self.__name_queue.append(seqname)
                self.__total_len += len(seq)
                self.__index_queue.append(self.__total_len)
                self.__id_queue.append(self.__nseqs)
                self.__nseqs += 1
                return seq
            except StopIteration:
                try:
                    self.__curr_iter = self.__read(next(self.__path_iter))
                except StopIteration:
                    self.__name_queue.append(None)
                    self.__index_queue.append(None)
                    self.__name_queue.append(None)
                    raise StopIteration()

    def __next__(self):
        ret = None
        if self.__curr_block_idx == self.__curr_block.shape[0]:
            # this raises the final StopIteration
            # when nothing is left to read
            seq = self.__read_next_seq()
            try:
                while len(seq) % 2 == 1:
                    tmp = DNA.concat([seq, self.__read_next_seq()])
                    seq = tmp
            except StopIteration:
                # there are no more files to read
                pass
            self.__curr_block, self.__padded = self.__pack(seq)
            self.__curr_block_idx = 0
        ret = self.__curr_block[self.__curr_block_idx]
        self.__curr_block_idx += 1
        return ret

    def __read(self, path):
        for seq_i, seq in enumerate(skbio.io.read(path, format='fasta',)):
            ltag = self.ltag_re.search(seq.metadata['description'])
            if ltag is None:
                ltags = str(seq_i)
            else:
                ltag = ltag.groups()[0]
            yield seq, ltag

    def __pack(self, seq):
        tfm = self.ohe.transform(seq.values.astype('U').reshape(-1,1)).T
        tfm[:self.nchars] += tfm[self.nchars:]
        tfm = tfm[:self.nchars]
        col_mask = tfm[self._col2drop].squeeze() == 1
        tfm = (tfm[self._row_mask]).astype(np.uint8)
        tfm[:,col_mask] = 1
        tfm = tfm.T
        padded = False
        if tfm.shape[0] % 2 == 1:
            padded = True
            tfm = np.append(tfm, [[0, 0, 0, 0]], axis=0)
        packed = np.packbits(np.concatenate((tfm[0::2], tfm[1::2]), axis=1))
        return packed, padded

class QueueIterator(object):

    def __init__(self, queue):
        self.__queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        ret = self.__queue.popleft()
        if ret is None:
            raise StopIteration()
        return ret

class SeqConcat(object):
    ltag_re = re.compile('\[locus_tag=([A-Za-z0-9_]+)\]')

    def __init__(self):
        self.dna_chars = 'ATCGN'

        chars = self.dna_chars
        chars = list(chars + chars.lower())
        self.nchars = len(chars)//2

        self.ohe = OneHotEncoder(sparse=False)
        self.ohe.fit(np.array(chars).reshape(-1, 1))
        categories = self.ohe.categories_[0][:self.nchars]

        self._col2drop = categories == 'N'
        self._row_mask = np.zeros(len(categories), dtype=bool)
        self._row_mask[np.logical_not(self._col2drop)] = True


    def _read(self, path):
        for seq_i, seq in enumerate(skbio.io.read(path, format='fasta',)):
            self._seqlens.append(len(seq))
            ltag = self.ltag_re.search(seq.metadata['description'])
            if ltag is None:
                self._ltags.append(str(seq_i))
            else:
                self._ltags.append(ltag.groups()[0])
            yield seq
        self._seqindex = np.cumsum(self._seqlens)
        self._ltags = np.array(self._ltags)

    def _read_path(self, path):
        self._seqlens = list()
        self._ltags = list()

        cat_seq = skbio.sequence.DNA.concat(self._read(path))

        tfm = self.ohe.transform(cat_seq.values.astype('U').reshape(-1,1)).T
        tfm[:self.nchars] += tfm[self.nchars:]
        tfm = tfm[:self.nchars]
        col_mask = tfm[self._col2drop].squeeze() == 1

        #tfm = (tfm[self._row_mask] * 4).astype(np.uint8)
        tfm = (tfm[self._row_mask]).astype(np.uint8)
        tfm[:,col_mask] = 1
        return tfm, self._seqindex, self._ltags

def pack_ohe_dna(ohe):
    """
    Pack a one-hot encoded DNA sequence
    """
    bits = (ohe.T != 0).astype(np.uint8)
    padded = False
    if bits.shape[0] % 2 == 1:
        padded = True
        bits = np.append(bits, [[0, 0, 0, 0]], axis=0)
    packed = np.packbits(np.concatenate((bits[0::2], bits[1::2]), axis=1))
    return packed, padded

def write_packed(path, raw, packed, seqindex, ltags, ckwargs=dict(compression='gzip')):
    print("writing packed data to", path)
    try:
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('packed', data=packed, dtype=np.uint8, **ckwargs)
            dset.attrs['padded'] = padded
            dset = f.create_dataset('packed_index', data=seqindex, **ckwargs)
            dset = f.create_dataset('ltags', shape=ltags.shape, dtype=h5py.special_dtype(vlen=str), **ckwargs)
            dset[:] = ltags
            dset = f.create_dataset('raw', data=raw.T, **ckwargs)
    except Exception as e:
        print("couldn't open", path)
        print(e.args[0])
    return os.path.getsize(path)

def read_h5_seq(path, read_ltags=True):
    try:
        with h5py.File(path, 'r') as f:
            dset = f['packed']
            seq = np.unpackbits(dset[:]).reshape(-1, 4)
            if dset.attrs['padded']:
                seq = seq[:-1]
            seq = seq.T
            index = f['packed_index'][:]
            ltags = None
            if read_ltags:
                ltags = f['ltags'][:]
        return seq, index, ltags
    except Exception as e:
        print("couldn't open", h5path)
        print(e.args[0])
        return None, None, None

def read_fna_seq(path):
    sc = SeqConcat()
    data, seqindex, ltags = sc._read_path(path)
    return data, seqindex, ltags


def get_seq_packed(packed, index, i):
    """
    Slice ragged array of *packed* one-hot encoded DNA sequence
    """
    start = 0 if i == 0 else index[i-1]
    end = index[i]
    shift = start % 2
    unpacked = np.unpackbits(packed[start//2:math.ceil(end/2)]).reshape(-1, 4)[shift:shift+end-start]
    return unpacked.T


def get_seq(binseq, index, i):
    """
    Slice ragged array of one-hot encoded DNA sequence
    """
    start = 0 if i == 0 else index[i-1]
    end = index[i]
    seq = binseq[start:end]
    return seq.T