#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""
import numpy as np
import unittest


from pax import pax
from pax.datastructure import Event, Peak, Waveform


class TestPosRecWeightedSum(unittest.TestCase):
    def setUp(self):
        self.conf = pax.get_configuration("")
        self.plugin_source = pax.get_plugin_source(self.conf)
        self.objy = pax.instantiate('PosSimple.PosRecWeightedSum',
                                    self.plugin_source,
                                    self.conf)

        self.e = Event()


        self.e.peaks.append(Peak({'left' : 5,
                                  'right' : 9}))

    def test_something(self):
        self.e.pmt_waveforms = np.arange(1000).reshape(100,10)
        e2 = self.objy.process_event(self.e)

        rps = e2.peaks[0].reconstructed_positions
        self.assertEqual(len(rps), 1)
        self.assertEqual(rps[0].algorithm, 'PosRecWeightedSum')

        self.assertAlmostEqual(rps[0].x, 0.012204295277433013)
        self.assertEqual(rps[0].y, -0.09300181089384905)
        self.assertNotEqual(rps[0].z, rps[0].z)  # nan



    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
