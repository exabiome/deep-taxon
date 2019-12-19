import skbio.io
from skbio.sequence import DNA, Protein
import re
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import deque
from abc import ABCMeta, abstractmethod


class AbstractSeqIterator(object, metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def characters(cls):
        pass

    @abstractmethod
    def pack(self, seq):
        pass

    ltag_re = re.compile('>lcl|([A-Za-z0-9_.]+)')

    def __init__(self, paths, logger=None, faa=False):
        self.logger = logger

        # setup our characters
        chars = self.characters()
        chars = list(chars.upper() + chars.lower())
        self.nchars = len(chars)//2

        self.ohe = OneHotEncoder(sparse=False)
        self.ohe.fit(np.array(chars).reshape(-1, 1))

        if isinstance(paths, str):
            paths = [paths]

        self.__paths = paths
        self.__path_iter = None
        self.__name_queue = None
        self.__index_queue = None
        self.__taxon_queue = None
        self.__id_queue = None
        self.__total_len = 0
        self.__nseqs = 0
        self.skbio_cls = Protein if faa else DNA
        self.logger.debug('reading %s' % self.skbio_cls.__name__)

        self.__curr_block = np.zeros((0, self.nchars), dtype=np.uint8)
        self.__curr_block_idx = 0

    @property
    def names(self):
        return self.__name_queue

    @property
    def taxon(self):
        return self.__taxon_queue

    @property
    def index(self):
        return self.__index_queue

    @property
    def ids(self):
        return self.__id_queue

    @property
    def index_iter(self):
        return QueueIterator(self.index, self)

    @property
    def names_iter(self):
        return QueueIterator(self.names, self)

    @property
    def id_iter(self):
        return QueueIterator(self.ids, self)

    @property
    def taxon_iter(self):
        return QueueIterator(self.taxon, self)

    def __iter__(self):
        self.__path_iter = iter(self.__paths)
        # initialize the sequence iterator
        self.__curr_iter = self.__read(next(self.__path_iter))
        self.__name_queue = deque()
        self.__index_queue = deque()
        self.__taxon_queue = deque()
        self.__id_queue = deque()
        self.__total_len = 0
        self.__nseqs = 0
        self.__curr_block = np.zeros((0, self.nchars), dtype=np.uint8)
        self.__curr_block_idx = 0
        self.__curr_file_idx = 0
        return self

    @property
    def total_len(self):
        return self.__total_len

    @property
    def curr_block_idx(self):
        return self.__curr_block_idx

    @property
    def curr_block(self):
        return self.__curr_block

    def __read_next_seq(self):
        while True:
            try:
                seq, seqname = next(self.__curr_iter)
                self.__name_queue.append(seqname)
                self.__taxon_queue.append(np.uint16(self.__curr_file_idx))
                self.__id_queue.append(self.__nseqs)
                self.__nseqs += 1
                return seq
            except StopIteration:
                try:
                    self.__curr_iter = self.__read(next(self.__path_iter))
                    self.__curr_file_idx += 1
                except StopIteration:
                    self.__name_queue.append(None)
                    self.__index_queue.append(None)
                    self.__taxon_queue.append(None)
                    self.__name_queue.append(None)
                    raise StopIteration()

    def __next__(self):
        self._load_buffer()
        ret = self.__curr_block[self.__curr_block_idx]
        self.__curr_block_idx += 1
        return ret

    def _load_buffer(self):
        if self.__curr_block_idx == self.__curr_block.shape[0]:
            # this raises the final StopIteration
            # when nothing is left to read
            seq = self.__read_next_seq()
            self.__curr_block = self.pack(seq)
            self.__total_len += self.__curr_block.shape[0]
            self.__index_queue.append(self.__total_len)
            self.__curr_block_idx = 0

    @classmethod
    def get_seqname(cls, seq):
        return seq.metadata['id']

    def __read(self, path):
        self.logger.debug('reading %s', path)
        kwargs = {'format': 'fasta', 'constructor': self.skbio_cls}
        for seq_i, seq in enumerate(skbio.io.read(path, **kwargs)):
            ltag = self.get_seqname(seq)
            yield seq, ltag


class DNASeqIterator(AbstractSeqIterator):

    @classmethod
    def characters(cls):
        return 'ATCGN'

    def __init__(self, paths, logger=None):
        super().__init__(paths, logger=logger, faa=False)

        categories = self.ohe.categories_[0][:self.nchars]

        self._col2drop = categories == 'N'
        self._row_mask = np.zeros(len(categories), dtype=bool)
        self._row_mask[np.logical_not(self._col2drop)] = True

    def pack(self, seq):
        tfm = self.ohe.transform(seq.values.astype('U').reshape(-1, 1)).T
        # the first half will be uppercase and the second half will be lowercase
        # - combine upper and lower
        tfm[:self.nchars] += tfm[self.nchars:]
        tfm = tfm[:self.nchars]

        col_mask = tfm[self._col2drop].squeeze() == 1
        tfm = (tfm[self._row_mask]).astype(np.uint8)
        tfm[:, col_mask] = 1
        tfm = tfm.T
        if tfm.shape[0] % 2 == 1:
            tfm = np.append(tfm, [[0, 0, 0, 0]], axis=0)
        packed = np.packbits(np.concatenate((tfm[0::2], tfm[1::2]), axis=1))
        return packed


class AASeqIterator(AbstractSeqIterator):

    aa_map = np.array([0]*65 + list(range(1, 27)), dtype=np.uint8)

    @classmethod
    def characters(cls):
        return 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, paths, logger=None):
        super().__init__(paths, logger=logger, faa=True)

    def pack(self, seq):
        nbits = len(seq)*5
        start = (8 - nbits % 8)
        nbits += start
        bits = np.zeros(nbits, dtype=np.uint8)
        s = start
        for i, aa in enumerate(str(seq)):
            num = self.aa_map[ord(aa)]
            e = s + 5
            bits[s:e] = np.unpackbits(num)[3:]
            s = e
        packed = np.packbits(bits)
        return packed


class QueueIterator(object):

    def __init__(self, queue, seqit):
        self.__queue = queue
        self.q = self.__queue
        self.__seqit = seqit
        self.sit = self.__seqit

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.__queue) == 0:
            self.__seqit._load_buffer()
        ret = self.__queue.popleft()
        if ret is None:
            raise StopIteration()
        return ret
