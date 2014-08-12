#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest


from pax import pax
from pax.datastructure import Event, Peak, Waveform


class TestPrintToScreen(unittest.TestCase):
    def setUp(self):
        conf = pax.get_configuration()
        plugin_source = pax.get_plugin_source(conf)
        self.obj = pax.instantiate('PrintToScreen.PrintToScreen',
                                   plugin_source,
                                   conf)

        self.e = Event()

    def test_something(self):
        self.obj.process_event(self.e)


if __name__ == '__main__':
    unittest.main()
