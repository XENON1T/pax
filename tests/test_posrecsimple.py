#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""
import unittest

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecWeightedSum(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={'pax': {
            'plugin_group_names': ['test'],
            'test':               'PosSimple.PosRecWeightedSum'}})
        self.posrec_plugin = self.pax.get_plugin_by_name('PosRecWeightedSum')

        self.e = Event()

        self.e.peaks.append(Peak({'left':  5,
                                  'right': 9,
                                  'type':  's2'}))

    def test_something(self):
        self.assertIsInstance(self.posrec_plugin, plugin.TransformPlugin)
        self.assertEqual(self.posrec_plugin.__class__.__name__, 'PosRecWeightedSum')
        pass
        # This test is broken: PosSimple doesn't use pmt_waveforms anymore, but area_per_pmt

        # self.e.pmt_waveforms = np.arange(1000).reshape(100, 10)
        # e2 = self.posrec_plugin.process_event(self.e)
        #
        # rps = e2.peaks[0].reconstructed_positions
        # self.assertEqual(len(rps), 1)
        # self.assertEqual(rps[0].algorithm, 'PosRecWeightedSum')
        #
        # self.assertAlmostEqual(rps[0].x, 0.012204295277433013)
        # self.assertEqual(rps[0].y, -0.09300181089384905)
        # self.assertNotEqual(rps[0].z, rps[0].z)  # nan

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
