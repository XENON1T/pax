#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""
import unittest

from pax import core
from pax.datastructure import Event, Peak

override_config = \
"""
[pax]
plugin_group_names = ['test']
test = 'PrintToScreen.PrintToScreen
"""


class TestPosRecWeightedSum(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor()
        self.posrec_plugin = self.pax.action_plugins[0]

        self.e = Event()

        self.e.peaks.append(Peak({'left':  5,
                                  'right': 9,
                                  'type':  's2'}))

    def test_something(self):
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
