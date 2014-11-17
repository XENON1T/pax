#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


from pax import core
from pax.datastructure import Event, Peak, Waveform


class TestPrintToScreen(unittest.TestCase):
    def setUp(self):
        self.obj = core.instantiate_plugin('PrintToScreen.PrintToScreen', for_testing=True)
        self.e = Event()

    def test_something(self):
        self.obj.process_event(self.e)


if __name__ == '__main__':
    unittest.main()
