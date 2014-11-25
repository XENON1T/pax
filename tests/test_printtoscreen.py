#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


from pax import core
from pax.datastructure import Event, Peak, Waveform


override_config = \
"""
[pax]
plugin_group_names = ['output']
output = 'PrintToScreen.PrintToScreen'
"""

class TestPrintToScreen(unittest.TestCase):
    def setUp(self):
        self.obj = override_config
        self.pax = core.Processor(config_names='XENON100', config_string=override_config, just_testing=True)
        self.obj = self.pax.action_plugins[-1]
        self.e = Event()

    def test_something(self):
        self.obj.process_event(self.e)


if __name__ == '__main__':
    unittest.main()
