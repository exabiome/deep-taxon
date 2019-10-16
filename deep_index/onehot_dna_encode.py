import skbio.io
import skbio.sequence
import re

class SeqConcat(object):
    ltag_re = re.compile('\[locus_tag=([A-Za-z0-9_]+)\]')

    def __init__(self):
        self.dna_chars = 'ATCGN'

        chars = self.dna_chars
        chars = list(chars + chars.lower())
        nchars = len(chars)//2

        self.ohe = OneHotEncoder(sparse=False)
        ohe.fit(np.array(chars).reshape(-1, 1))
        categories = ohe.categories_[0][:nchars]

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

        cat_seq = skbio.sequence.DNA.concat(sc._read(path))

        tfm = ohe.transform(cat_seq.values.astype('U').reshape(-1,1)).T
        tfm[:nchars] += tfm[nchars:]
        tfm = tfm[:nchars]
        col_mask = tfm[col2drop].squeeze() == 1

        tfm = (tfm[row_mask] * 4).astype(np.uint8)
        tfm[:,col_mask] = 1
        return tfm, self._seqindex, self._ltags



