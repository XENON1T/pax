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
        conf = pax.get_configuration("")
        plugin_source = pax.get_plugin_source(conf)
        self.obj = pax.instantiate('PosSimple.PosRecWeightedSum',
                                   plugin_source,
                                   conf)
        print(type(self.obj))


        #self.e.pmt_waveforms = np.arange(100).reshape(10,10)

        #self.e.peaks.append(Peak({'left' : 5,
        #                        'right' : 9}))

    def test_something(self):
        print('ho', type(self.obj), )
        self.assertEqual(self.obj.say_hi(), 1)
        e2 = self.obj.transform_event(Event())
        print(self.e.do_dict(), e2.to_dict())


    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
