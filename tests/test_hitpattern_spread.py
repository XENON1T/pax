import unittest
import numpy as np

from pax import core, plugin, units
from pax.datastructure import Peak, Event


class TestComputeComputePeakProperties(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={'pax': {
            'plugin_group_names': ['test'],
            'test':               'ComputePeakProperties.HitpatternSpread'}})
        self.plugin = self.pax.get_plugin_by_name('HitpatternSpread')

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')

    @staticmethod
    def example_event(channels_with_something):
        bla = np.zeros(242)
        bla[np.array(channels_with_something)] = 1
        e = Event.empty_event()
        e.peaks.append(Peak({'left':  5,
                             'right': 9,
                             'type':  'unknown',
                             'detector':  'tpc',
                             'area_per_channel': bla,
                             }))
        return e

    def test_get_plugin(self):
        self.assertIsInstance(self.plugin, plugin.TransformPlugin)
        self.assertEqual(self.plugin.__class__.__name__, 'HitpatternSpread')

    def test_compute_spread(self):
        e = self.example_event([1, 16])
        e = self.plugin.transform_event(e)
        self.assertIsInstance(e, Event)
        self.assertEqual(len(e.peaks), 1)
        p = e.peaks[0]

        # PMT 1 and 16 are aligned in y, 166.84 mm from center in x on opposite sides
        self.assertAlmostEqual(p.top_hitpattern_spread, 166.84 * units.mm)

        # If no hits, hitpattern spread should be nan
        self.assertEqual(p.bottom_hitpattern_spread, 0)
