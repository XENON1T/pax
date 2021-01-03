import unittest
import numpy as np

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecTopPatternFunctionFit(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON1T',
                                  just_testing=True,
                                  config_dict={'pax': {'plugin_group_names': ['test'],
                                                       'look_for_config_in_runs_db': False,
                                                       'test': ['WeightedSum.PosRecWeightedSum',
                                                                'TopPatternFit.PosRecTopPatternFunctionFit'],
                                                       'logging_level': 'debug'}})
        self.guess_plugin = self.pax.get_plugin_by_name('PosRecWeightedSum')
        self.plugin = self.pax.get_plugin_by_name('PosRecTopPatternFunctionFit')

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')
        delattr(self, 'guess_plugin')

    @staticmethod
    def example_event():
        top_hits = [7, 8, 8, 5, 8, 10, 6, 9, 3, 7, 6, 4, 5, 2, 1, 0, 7, 1, 3, 1, 4, 2, 5, 1, 4, 3,
                    1, 3, 2, 4, 3, 0, 4, 4, 1, 6, 2, 4, 9, 12, 8, 10, 9, 6, 9, 1, 2, 1, 2, 1, 4, 10,
                    0, 0, 1, 2, 1, 0, 2, 3, 6, 1, 3, 2, 3, 5, 2, 6, 30, 18, 24, 10, 8, 3, 4, 2, 4, 2,
                    1, 4, 3, 4, 5, 5, 2, 1, 2, 2, 2, 4, 12, 48, 139, 89, 19, 9, 3, 4, 2, 3, 1, 1, 6,
                    0, 3, 1, 2, 4, 12, 97, 87, 15, 6, 3, 4, 4, 0, 2, 3, 6, 13, 21, 3, 4, 3, 1, 7]
        hits = np.append(top_hits, np.zeros(254 - 127))
        e = Event.empty_event()
        e.peaks.append(Peak({'left':  5,
                             'right': 9,
                             'type':  'S2',
                             'detector':  'tpc',
                             'area': 123,
                             'area_per_channel': hits}))
        return e

    def test_posrec(self):
        self.assertIsInstance(self.plugin, plugin.TransformPlugin)
        self.assertEqual(self.plugin.__class__.__name__, 'PosRecTopPatternFunctionFit')

        e = self.example_event()
        e = self.guess_plugin.transform_event(e)
        e = self.plugin.transform_event(e)
        self.assertIsInstance(e, Event)
        self.assertEqual(len(e.peaks), 1)
        self.assertEqual(len(e.S2s()), 1)
        self.assertEqual(len(e.peaks[0].reconstructed_positions), 2)
        rp = e.peaks[0].reconstructed_positions[1]
        self.assertEqual(rp.algorithm, 'PosRecTopPatternFunctionFit')
        x_truth = 11.0882
        y_truth = 18.7855
        self.assertAlmostEqual(rp.x, x_truth, delta=3)
        self.assertAlmostEqual(rp.y, y_truth, delta=3)

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
