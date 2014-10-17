#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest


from pax import core
from pax.datastructure import Event, Peak, Waveform


class TestPrintToScreen(unittest.TestCase):
    def setUp(self):
        conf = core.init_configuration(config_names='XENON100')
        plugin_source = core.get_plugin_source(conf)
        self.obj = core.instantiate_plugin('PrintToScreen.PrintToScreen',
                                   plugin_source,
                                   conf)

        self.e = Event()

    def test_something(self):
        self.obj.process_event(self.e)


if __name__ == '__main__':
    unittest.main()
