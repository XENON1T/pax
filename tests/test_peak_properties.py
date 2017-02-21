from __future__ import division
import unittest
import numpy as np
from numpy import testing as np_testing

from pax.plugins.peak_processing.BasicProperties import integrate_until_fraction, put_w_in_center_of_field


class TestPeakProperties(unittest.TestCase):

    def test_integrate_until_fraction(self):
        # Test a simple ones-only waveform, for which no interpolation will be needed
        w = np.ones(100, dtype=np.float32)
        fractions_desired = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64) / 100
        result = np.zeros(len(fractions_desired))
        integrate_until_fraction(w, fractions_desired, result)
        np_testing.assert_almost_equal(result, fractions_desired * 100, decimal=4)

        # Now test a one-sample waveform, which will probe the interpolation stuff
        w = np.ones(1, dtype=np.float32)
        result = np.zeros(len(fractions_desired))
        integrate_until_fraction(w, fractions_desired, result)
        np_testing.assert_almost_equal(result, fractions_desired, decimal=4)

    def test_store_waveform(self):
        field = np.zeros(5)
        put_w_in_center_of_field(np.ones(3), field, 0)
        np_testing.assert_equal(field, np.array([0, 0, 1, 1, 1]))

        field = np.zeros(5)
        put_w_in_center_of_field(np.ones(3), field, 1)
        np_testing.assert_equal(field, np.array([0, 1, 1, 1, 0]))

        field = np.zeros(5)
        put_w_in_center_of_field(np.ones(3), field, 2)
        np_testing.assert_equal(field, np.array([1, 1, 1, 0, 0]))

        # Left overhang
        field = np.zeros(5)
        put_w_in_center_of_field(np.ones(4), field, 3)
        np_testing.assert_equal(field, np.array([1, 1, 1, 0, 0]))

        field = np.zeros(5)
        put_w_in_center_of_field(np.ones(7), field, 6)
        np_testing.assert_equal(field, np.array([1, 1, 1, 0, 0]))

        # Right overhang
        field = np.zeros(5)
        put_w_in_center_of_field(np.ones(4), field, 0)
        np_testing.assert_equal(field, np.array([0, 0, 1, 1, 1]))

        field = np.zeros(5)
        put_w_in_center_of_field(np.ones(7), field, 0)
        np_testing.assert_equal(field, np.array([0, 0, 1, 1, 1]))

        # Waveform larger than field
        field = np.zeros(5)
        put_w_in_center_of_field(np.ones(20), field, 10)
        np_testing.assert_equal(field, np.array([1, 1, 1, 1, 1]))


if __name__ == '__main__':
    unittest.main()
