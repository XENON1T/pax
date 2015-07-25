import unittest
import numpy as np

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecRobustWeightedmean(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={'pax': {
            'plugin_group_names': ['test'],
            'test':               'RobustWeightedMean.PosRecRobustWeightedMean'}})
        self.plugin = self.pax.get_plugin_by_name('PosRecRobustWeightedMean')

    @staticmethod
    def example_event(channels_with_something, area_per_channel=1):
        bla = np.zeros(243)
        bla[np.array(channels_with_something)] = area_per_channel
        e = Event.empty_event()
        e.peaks.append(Peak({'left':  5,
                             'right': 9,
                             'type':  'S2',
                             'detector':  'tpc',
                             'area_per_channel': bla}))
        return e

    def test_get_maxpmt_plugin(self):
        self.assertIsInstance(self.plugin, plugin.TransformPlugin)
        self.assertEqual(self.plugin.__class__.__name__, 'PosRecRobustWeightedMean')

    def test_posrec(self):
        """Test a hitpattern with an outlier"""
        e = self.example_event(channels_with_something=[8, 89, 91, 95, 93, 97, 98],
                               area_per_channel=1)
        e = self.plugin.transform_event(e)
        self.assertIsInstance(e, Event)
        self.assertEqual(len(e.peaks), 1)
        self.assertEqual(len(e.S2s()), 1)
        self.assertEqual(len(e.peaks[0].reconstructed_positions), 1)
        rp = e.peaks[0].reconstructed_positions[0]
        self.assertEqual(rp.algorithm, self.plugin.name)
        self.assertAlmostEqual(rp.x, 0, delta=0.2)
        self.assertAlmostEqual(rp.y, 0, delta=0.2)

if __name__ == '__main__':
    unittest.main()
