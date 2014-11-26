#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


from pax import core
from pax.datastructure import Event, Peak, Waveform



class TestPrintToScreen(unittest.TestCase):
    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={'pax': {
            'plugin_group_names': ['output'],
            'output':              'PrintToScreen.PrintToScreen'}})
        self.obj = self.pax.get_plugin_by_name('PrintToScreen')
        self.e = Event()

    def test_something(self):
        self.obj.process_event(self.e)


if __name__ == '__main__':
    unittest.main()
