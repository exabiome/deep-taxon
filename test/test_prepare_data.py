import numpy as np
import skbio.stats.distance as ssd
from exabiome.gtdb.prepare_data import get_nonrep_matrix

def get_toy_dmat():
    """
    Get a simple skbio DistanceMatrix for testing
    """
    dist = np.array(
        [ [0, 1, 2],
          [1, 0, 3],
          [2, 3, 0] ]
    )
    ids = ['a', 'b', 'c']
    toy_dmat = ssd.DistanceMatrix(dist, ids=ids)
    return toy_dmat


def test_rep_only():
    """
    Test getting altered distance matrix when
    query only contains representative species
    """
    toy_dmat = get_toy_dmat()
    dmat = get_nonrep_matrix(['b', 'c', 'a'],
                             ['b', 'c', 'a'], toy_dmat)
    expected = np.array([[0., 3., 1.],
                         [3., 0., 2.],
                         [1., 2., 0.]])
    np.testing.assert_array_equal(expected, dmat.data)


def test_nonrep():
    """
    Test getting distance matrix with non-reprsentative
    species from a distance matrix for representative
    species
    """
    toy_dmat = get_toy_dmat()
    expected = np.array([[0., 3., 0., 1.],
                         [3., 0., 3., 2.],
                         [0., 3., 0., 1.],
                         [1., 2., 1., 0.]])
    dmat = get_nonrep_matrix(['b', 'c', 'd', 'a'],
                             ['b', 'c', 'b', 'a'], toy_dmat)
    np.testing.assert_array_equal(expected, dmat.data)
