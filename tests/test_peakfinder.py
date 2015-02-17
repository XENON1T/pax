import unittest
import numpy as np

from pax import core, plugin, units
from pax.datastructure import Occurrence, Event
from pax.utils import empty_event

# 100 normal(0,1)'s
example_noise = np.array([
    0.27115415,  1.18243054,  0.0401233,  2.53551981,  1.34943168,
    0.29614701, -0.26284844,  0.42906156, -0.60732569, -0.76611985,
    2.0399306, -0.89244998,  0.40748667,  0.71946219,  1.1534543,
    0.36952817, -0.40280752,  0.2779532,  0.25917277,  2.48082239,
    1.14793643,  1.16925813,  1.34584534,  1.55723492,  0.27005464,
    -0.70915584,  0.99380989, -0.41031646, -0.49883865, -1.97879413,
    0.92993994, -0.66888725, -1.23772836,  0.12832938,  0.04884238,
    1.41350955, -0.00846222,  0.0384421,  1.30722541, -0.65153134,
    0.39958294, -1.00408831,  0.70918933, -1.19184558,  1.05374838,
    -0.56706629,  0.69658105,  0.46209854, -0.19891502, -0.35991869,
    0.63242906, -0.13912786,  0.36476287,  0.35366695,  0.12790934,
    0.38541716,  2.63752326,  0.35874707, -0.4328674, -1.92123614,
    -0.78920336,  0.75493152, -1.12784897,  1.39582886,  0.88185902,
    0.74021747,  0.05429847,  1.04589925,  1.13863526,  0.76766802,
    0.07443457, -0.42795935,  1.37122591, -1.48361599,  2.09405696,
    0.98183696,  0.7484914,  1.97241525, -1.74060046, -0.12082176,
    -0.03596427, -0.04222192,  1.11793958,  0.48307736,  0.49251086,
    -1.3183425,  0.45331936,  1.1363642,  0.41380787, -2.07808598,
    0.39172201,  0.93118465,  2.33169722, -1.35306737, -1.62536499,
    -0.93964013,  0.18316281, -2.79421714, -0.43965162,  0.55478247
], dtype=np.float64)


class TestSmallPeakfinder(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={
            'pax': {
                'plugin_group_names': ['test'],
                'test':               'SmallPeakfinder.FindSmallPeaks',
                # 'logging_level':      'DEBUG',
                },
            'DEFAULT': {
                'gains': [1, 1],
                'channels_top': [0],
                'channels_bottom': [1],
                'channels_in_detector': {'tpc': [0, 1]}}})
        self.plugin = self.pax.get_plugin_by_name('FindSmallPeaks')

    @staticmethod
    def peak_at(left, right, amplitude, noise_sigma):
        w = example_noise * noise_sigma
        w[left:right + 1] = amplitude
        return w

    def test_sanity(self):
        self.assertIsInstance(self.plugin, plugin.TransformPlugin)
        self.assertEqual(self.plugin.__class__.__name__, 'FindSmallPeaks')

    def try_single_clear_peak(self, left, right):
        raw_peaks = np.zeros((100, 2), dtype=np.int64)
        waveform = self.peak_at(left, right, amplitude=100, noise_sigma=0.05)

        n_found = self.plugin._numba_find_peaks(waveform, float(1), float(3), raw_peaks)
        self.assertEqual(n_found, 1)
        self.assertEqual(raw_peaks[0, 0], left)
        self.assertEqual(raw_peaks[0, 1], right)

        mean_std_result = np.array([0, 0], dtype=np.float64)
        self.plugin._numba_mean_std_outside_peaks(waveform, raw_peaks, mean_std_result)

        mask = np.ones(100, dtype=np.bool)
        mask[left:right + 1] = False
        mean_should_be = np.mean(waveform[mask])
        self.assertAlmostEqual(mean_std_result[0], mean_should_be)
        std_should_be = np.std(waveform[mask])
        self.assertAlmostEqual(mean_std_result[1], std_should_be)

    def test_single_peaks(self):
        # 10 samples wide
        self.try_single_clear_peak(10, 20)
        self.try_single_clear_peak(0, 20)
        self.try_single_clear_peak(80, 99)

        # 2 samples wide
        self.try_single_clear_peak(5, 6)
        self.try_single_clear_peak(0, 1)
        self.try_single_clear_peak(98, 99)

        # 1 sample wide
        self.try_single_clear_peak(5, 5)
        self.try_single_clear_peak(0, 0)
        self.try_single_clear_peak(99, 99)


if __name__ == '__main__':
    unittest.main()
