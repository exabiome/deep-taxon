from scipy.spatial.distance import squareform
import torch
import torch.nn.functional as F

class SplitCollater:

    def __init__(self, padval, freq=1.0, factors=[2, 4, 8]):
        self.padval = padval
        self.freq = freq
        self.factors = factors

    def __call__(self, samples):
        maxlen = 0
        l_idx = -1
        if isinstance(samples, tuple):
            samples = [samples]
        for i, X, y, seq_id, genome in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]
        X_ret = list()
        y_ret = list()
        idx_ret = list()
        size_ret = list()
        seq_id_ret = list()
        genome_ret = list()
        for i, X, y, seq_id, genome in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif), value=self.padval)
            X_ret.append(X_)
            y_ret.append(y)
            size_ret.append(X.shape[l_idx])
            idx_ret.append(i)
            seq_id_ret.append(seq_id)
            genome_ret.append(genome)
        X_ret = torch.stack(X_ret)
        y_ret = torch.stack(y_ret)
        size_ret = torch.tensor(size_ret)
        idx_ret = torch.tensor(idx_ret)
        seq_id_ret = torch.tensor(seq_id_ret)
        genome_ret = torch.tensor(genome_ret)

        if self.freq == 1.0 or rs.rand() < self.freq:
            f = self.factors[rs.randint(len(self.factors))]
            y_ret = y_ret.repeat_interleave(f)
            seq_id_ret = seq_id_ret.repeat_interleave(f)
            idx_ret = idx_ret.repeat_interleave(f)
            X_ret = X_ret.reshape((X_ret.shape[0] * f, X_ret.shape[0] // f))

            q = lens // f
            r = lens // f
            n_bad_chunks = b.shape[-1]//f - q
            bad_chunk_pos = torch.where(n_bad_chunks > 0)[0]
            start_bad_chunks = bad_chunk_pos + q[bad_chunk_pos]

        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret, genome_ret)


class SeqCollater:

    def __init__(self, padval):
        self.padval = padval

    def __call__(self, samples):
        maxlen = 0
        l_idx = -1
        if isinstance(samples, tuple):
            samples = [samples]
        for i, X, y, seq_id, genome in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]
        X_ret = list()
        y_ret = list()
        idx_ret = list()
        size_ret = list()
        seq_id_ret = list()
        genome_ret = list()
        for i, X, y, seq_id, genome in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif), value=self.padval)
            X_ret.append(X_)
            y_ret.append(y)
            size_ret.append(X.shape[l_idx])
            idx_ret.append(int(i))
            seq_id_ret.append(int(seq_id))
            genome_ret.append(int(genome))
        X_ret = torch.stack(X_ret)
        y_ret = torch.stack(y_ret)
        size_ret = torch.tensor(size_ret)
        idx_ret = torch.tensor(idx_ret)
        seq_id_ret = torch.tensor(seq_id_ret)
        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


class TrainingSeqCollater:

    def __init__(self, padval, seq_dtype=torch.uint8):
        self.padval = padval
        self.seq_dtype = seq_dtype

    def __call__(self, samples):
        maxlen = 0
        l_idx = -1
        if isinstance(samples, tuple):
            samples = [samples]
        for i, X, y, seq_id, genome_id in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]
        X_ret = list()
        y_ret = list()
        for i, X, y, seq_id, genome_id in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif), value=self.padval)
            X_ret.append(X_)
            y_ret.append(y)
        X_ret = torch.stack(X_ret, out=torch.zeros(len(X_ret), len(X_ret[0]), dtype=self.seq_dtype))
        y_ret = torch.stack(y_ret, out=torch.zeros(len(X_ret), dtype=torch.int64))
        return (X_ret, y_ret)


def _check_collater(padval, seq_collater):
    if seq_collater is not None:
        return seq_collater
    elif padval is not None:
        return SeqCollater(padval)
    else:
        raise ValueError("must specify padval or seq_collater")

class GraphCollater:

    def __init__(self, node_ids, padval=None, seq_collater=None):
        self.collater = _check_collater(padval, seq_collater)
        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long)

    def __call__(self, samples):
        """
        A function to collate samples and return a sub-distance matrix
        """
        idx_ret, X_ret, y_idx, size_ret, seq_id_ret = self.collater(samples)

        # Get distances
        y_ret = self.node_ids[y_idx]
        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


class DistanceCollater:

    def __init__(self, dmat, padval=None, seq_collater=None):
        self.collater = _check_collater(padval, seq_collater)
        if len(dmat.shape) == 1:
            dmat = squareform(dmat)
        #self.dmat = torch.as_tensor(dmat, dtype=torch.float).pow(2)
        #self.dmat = torch.as_tensor(dmat, dtype=torch.float).sqrt()
        self.dmat = dmat
        if not isinstance(self.dmat, torch.Tensor):
            self.dmat = torch.from_numpy(self.dmat)

        self.dmat /= self.dmat.max()


    def __call__(self, samples):
        """
        A function to collate samples and return a sub-distance matrix
        """
        X_ret, y_idx = self.collater(samples)

        # Get distances
        y_idx = y_idx.long()
        y_ret = self.dmat[y_idx][:, y_idx]
        return X_ret, y_ret


class TnfCollater:
    def __init__(self, vocab):
        self.bases = 4**torch.arange(4)
        rcmap = torch.tensor([3, 2, 1, 0])
        canonical = list()
        noncanonical = list()
        palindromes = list()
        seen = torch.zeros(256, dtype=bool)
        for i in range(256):
            if seen[i]:
                continue
            ar = torch.zeros(4, dtype=int)
            ar[3], r = divmod(i, 64)
            ar[2], r = divmod(r, 16)
            ar[1], ar[0] = divmod(r, 4)
            rc = rcmap[ar.flip(0)]
            rc_i = rc.matmul(self.bases)
            if i < rc_i:
                canonical.append(i)
                noncanonical.append(rc_i)
            elif rc_i < i:
                canonical.append(rc_i)
                noncanonical.append(i)
            else:
                palindromes.append(i)
            seen[i] = True
            seen[rc_i] = True
        self.canonical = torch.tensor(canonical)
        self.noncanonical = torch.tensor(noncanonical)
        self.palindromes = torch.tensor(palindromes)

        # calculate a map to convert DNA characters into 0-4 encoding
        self.cmap = torch.zeros(128, dtype=int) - 1
        count = 0
        self.padval = None
        for i, c in enumerate(vocab):
            if c == 'A':
                self.cmap[i] = 0            # A
                count += 1
            elif c == 'T':
                self.cmap[i] = 3            # T
                count += 1
            elif c == 'C':
                self.cmap[i] = 1            # C
                count += 1
            elif c == 'G':
                self.cmap[i] = 2            # G
                count += 1
            elif c == 'N':
                self.padval = i
                count += 1
            if count == 5:
                break
        if self.padval is None:
            raise ValueError("Could not find 'N' character in vocab -- this is needed to pad sequences")


    def __call__(self, samples):
        l_idx = -1
        if isinstance(samples, tuple):
            samples = [samples]

        maxlen = 0
        for i, X, y, seq_id, genome in samples:
            if maxlen < X.shape[l_idx]:
                maxlen = X.shape[l_idx]

        X_ret = list()
        y_ret = list()
        idx_ret = list()
        size_ret = list()
        seq_id_ret = list()
        genome_ret = list()
        for i, X, y, seq_id, genome in samples:
            dif = maxlen - X.shape[l_idx]
            X_ = X
            if dif > 0:
                X_ = F.pad(X, (0, dif), value=self.padval)
            X_ret.append(X_)
            y_ret.append(y)
            size_ret.append(X.shape[l_idx])
            idx_ret.append(i)
            seq_id_ret.append(seq_id)
            genome_ret.append(genome)

        # calculate tetranucleotide frequency
        chunks = torch.stack(X_ret)

        ## 1. hash 4-mers
        __seq = self.cmap[chunks]
        i4mers = torch.stack([__seq[:, 0:-3], __seq[:, 1:-2], __seq[:, 2:-1], __seq[:, 3:]], axis=2)
        mask = torch.any(i4mers < 0, axis=2)
        h4mers = i4mers.matmul(self.bases)       # hashed 4-mers
        h4mers[mask] = 256    # use 257 to mark any 4-mers that had ambiguous nucleotides

        ## 2. count hashed 4-mers i.e. count integers from between 0-257 inclusive
        tnf = torch.zeros((32, 257), dtype=float)
        for i in range(tnf.shape[0]):
            counts = torch.bincount(h4mers[i], minlength=257)
            tnf[i] = counts/i4mers.shape[1]

        ## 3. merge canonical 4-mers
        canon_tnf = torch.zeros((32, 136))
        canon_tnf[:, :len(self.canonical)] = tnf[:, self.canonical] + tnf[:, self.noncanonical]
        canon_tnf[:, len(self.canonical):] = tnf[:, self.palindromes]

        X_ret = canon_tnf
        y_ret = torch.stack(y_ret)
        size_ret = torch.tensor(size_ret)
        idx_ret = torch.tensor(idx_ret)
        seq_id_ret = torch.tensor(seq_id_ret)

        return (idx_ret, X_ret, y_ret, size_ret, seq_id_ret)


def get_collater(dataset, inference=False, condensed=False):
    if dataset.tnf:
        return TnfCollater(dataset.vocab)
    elif dataset.manifold:
        if condensed:
            return TrainingSeqCollater(dataset.padval)
        else:
            return DistanceCollater(dataset.difile.distances, seq_collater=TrainingSeqCollater(dataset.padval))
    elif dataset.graph:
        return GraphCollater(dataset.difile.node_ids, seq_collater=TrainingSeqCollater(dataset.padval))
    else:
        if inference:
            return SeqCollater(dataset.padval)
        else:
            return TrainingSeqCollater(dataset.padval)
