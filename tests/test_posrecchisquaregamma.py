import unittest
import numpy as np

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecChiSquareGamma(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100',
                                  just_testing=True,
                                  config_dict={'pax': {'plugin_group_names': ['test'],
                                                       'test': 'PosRecChiSquareGamma.PosRecChiSquareGamma',
                                                       'logging_level': 'debug'}})
        self.plugin = self.pax.get_plugin_by_name('PosRecChiSquareGamma')

        self.e = Event.empty_event()

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')

    def example_event(self, channels_with_something):
        channels = np.array(channels_with_something, dtype='float64')
        e = Event.empty_event()
        e.peaks.append(Peak({'left':  5,
                             'right': 9,
                             'type':  'S2',
                             'detector':  'tpc',
                             'area': 123,
                             'area_per_channel': channels}))
        return e

    def test_posrec(self):
        self.assertIsInstance(self.plugin, plugin.TransformPlugin)
        self.assertEqual(self.plugin.__class__.__name__, 'PosRecChiSquareGamma')

        e = self.example_event([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                0, 1, 1, 0, 0, 3, 6, 7, 2, 2, 0, 0, 0, 0, 0,
                                1, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 2,
                                3, 2, 14, 33, 12, 5, 2, 2, 0, 1, 0, 0, 1, 1,
                                0, 1, 0, 0, 0, 0, 0, 0, 5, 40, 226, 45, 7, 0,
                                2, 1, 1, 0, 1, 0, 0, 0, 1, 0, 7, 14, 36, 3, 0,
                                2, 1, 0, 0, 0, 1, 4, 4, 3, 1, 0])
        e = self.plugin.transform_event(e)
        self.assertIsInstance(e, Event)
        self.assertEqual(len(e.peaks), 1)
        self.assertEqual(len(e.S2s()), 1)
        self.assertEqual(len(e.peaks[0].reconstructed_positions), 1)
        rp = e.peaks[0].reconstructed_positions[0]
        self.assertEqual(rp.algorithm, 'PosRecChiSquareGamma')
        # position agrees within one cm for test
        x_truth = 3.1625
        y_truth = -10.0172
        self.assertAlmostEqual(rp.x, x_truth, delta=1)
        self.assertAlmostEqual(rp.y, y_truth, delta=1)

if __name__ == '__main__':
    unittest.main()
