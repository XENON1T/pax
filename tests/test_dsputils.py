import numpy as np
import unittest

from pax import dsputils


class TestDSPUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_where_changes(self):
        example = np.array([0, 1, 2, 0, 4, -1, 60, 700, -4])
        b_t, b_f = dsputils.where_changes(example > 0)
        np.testing.assert_array_equal(b_f, np.array([3, 5, 8]))
        np.testing.assert_array_equal(b_t, np.array([1, 4, 6]))


    # TODO: test dsputils.intervals_where, others