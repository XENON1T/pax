import unittest
import numpy as np

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecTopPatternFit(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100',
                                  just_testing=True,
                                  config_dict={'pax': {'plugin_group_names': ['test'],
                                                       'test': ['WeightedSum.PosRecWeightedSum',
                                                                'TopPatternFit.PosRecTopPatternFit'],
                                                       'logging_level': 'debug'}})
        self.guess_plugin = self.pax.get_plugin_by_name('PosRecWeightedSum')
        self.plugin = self.pax.get_plugin_by_name('PosRecTopPatternFit')

        self.e = Event.empty_event()

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')
        delattr(self, 'guess_plugin')

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
        self.assertEqual(self.plugin.__class__.__name__, 'PosRecTopPatternFit')

        e = self.example_event([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                0, 1, 1, 0, 0, 3, 6, 7, 2, 2, 0, 0, 0, 0, 0,
                                1, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 2,
                                3, 2, 14, 33, 12, 5, 2, 2, 0, 1, 0, 0, 1, 1,
                                0, 1, 0, 0, 0, 0, 0, 0, 5, 40, 226, 45, 7, 0,
                                2, 1, 1, 0, 1, 0, 0, 0, 1, 0, 7, 14, 36, 3, 0,
                                2, 1, 0, 0, 0, 1, 4, 4, 3, 1, 0])
        e = self.guess_plugin.transform_event(e)
        e = self.plugin.transform_event(e)
        self.assertIsInstance(e, Event)
        self.assertEqual(len(e.peaks), 1)
        self.assertEqual(len(e.S2s()), 1)
        self.assertEqual(len(e.peaks[0].reconstructed_positions), 2)
        rp = e.peaks[0].reconstructed_positions[1]
        self.assertEqual(rp.algorithm, 'PosRecTopPatternFit')
        # position agrees within one cm for test
        x_truth = 3.1625
        y_truth = -10.0172
        self.assertAlmostEqual(rp.x, x_truth, delta=0.3)
        self.assertAlmostEqual(rp.y, y_truth, delta=0.3)

        try:
            from matplotlib import _cntr
        except ImportError:
            # Cannot test confidence tuple generation
            pass
        else:
            cts = rp.confidence_tuples
            self.assertEqual(len(cts), 2)


if __name__ == '__main__':
    unittest.main()
