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

        print(self.e, e2.to_json())



    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
