import unittest
import numpy as np

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecMaxPMT(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={'pax': {
            'plugin_group_names': ['test'],
            'test':               'MaxPMT.PosRecMaxPMT'}})
        self.plugin = self.pax.get_plugin_by_name('PosRecMaxPMT')

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')

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
        self.assertEqual(self.plugin.__class__.__name__, 'PosRecMaxPMT')

    def test_posrec(self):
        """Test a hitpattern of all ones and one 2 (at PMT 42)"""
        ch = 42     # Could test more locations, little point
        hitp = np.ones(len(self.plugin.config['channels_top']))
        hitp[ch] = 2
        e = self.example_event(channels_with_something=self.plugin.config['channels_top'],
                               area_per_channel=hitp)
        e = self.plugin.transform_event(e)
        self.assertIsInstance(e, Event)
        self.assertEqual(len(e.peaks), 1)
        self.assertEqual(len(e.S2s()), 1)
        self.assertEqual(len(e.peaks[0].reconstructed_positions), 1)
        rp = e.peaks[0].reconstructed_positions[0]
        self.assertEqual(rp.algorithm, self.plugin.name)
        self.assertEqual(rp.x, self.plugin.config['pmts'][ch]['position']['x'])
        self.assertEqual(rp.y, self.plugin.config['pmts'][ch]['position']['y'])


if __name__ == '__main__':
    unittest.main()
