import unittest
from exabiome.sequence import SequenceTable, VocabData, TaxaTable


class TaxaTableTest(unittest.TestCase):

    @classmethod
    def get_table(cls):
        kwargs = {
            'name':         'taxa_table',
            'description':  'a test TaxaTable',
            'taxon_id' :    ['TAX1', 'TAX2'],
            'embedding' :   [[0.1, 0.2], [0.3, 0.4]],
            'phylum' :      ['p1', 'p2'],
            'class' :       ['c1', 'c2'],
            'order' :       ['o1', 'o2'],
            'family' :      ['f1', 'f2'],
            'genus' :       ['g1', 'g2'],
            'species' :     ['s1', 's2']
        }
        return TaxaTable(**kwargs)

    def test_constructor(self):
        taxa_table = self.get_table()


class SequenceTableTest(unittest.TestCase):

    @classmethod
    def get_table(cls):
        taxa_table = TaxaTableTest.get_table()
        sequence = 'AATCGATCGGGGGCTAAGCCTACACATG'
        kwargs = {
            'name':             'sequence_table',
            'description':      'a test sequence table',
            'sequence_name':    ['seq1', 'seq2', 'seq3', 'seq4', 'seq5'],
            'sequence':         [0, 0, 3, 1, 2, 0, 3, 1, 2, 2, 2, 2, 2, 1,
                                 3, 0, 0, 2, 1, 1, 3, 0, 1, 0, 1, 0, 3, 2],
            'sequence_index':   [6, 14, 20, 24, 28],
            'length':           [5, 8, 6, 4, 6],
            'taxon':            [0, 0, 1, 1, 1],
            'taxon_table':      taxa_table,
            'vocab':            'dna',
        }
        table = SequenceTable(**kwargs)
        return table, sequence

    def test_constructor(self):
        table, sequence = self.get_table()

    def test_vocab_resolution(self):
        table, sequence = self.get_table()
        seq_col = table['sequence'].target
        seq_from_col = "".join(seq_col[i] for i in range(len(seq_col)))
        self.assertEqual(seq_from_col, sequence)

    def test_vocab_resolution(self):
        table, sequence = self.get_table()
        seq_col = table['sequence']
        seqs = [seq_col[i] for i in range(len(seq_col))]
        expected = ['AATCGA', 'TCGGGGGC', 'TAAGCC', 'TACA', 'CATG']
        self.assertListEqual(seqs, expected)
