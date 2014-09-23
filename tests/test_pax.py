#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""

import unittest

from pax import units
from pax import core


class TestPax(unittest.TestCase):

    def setUp(self):
        self.basic_config_header = "[pax]\nfinal_ancestor = True\n"
        pass

    def test_evaluate_configuration_string(self):
        x = self.basic_config_header + "test: \"mystring\""
        y = core.parse_configuration_string(x)
        self.assertEqual(y['pax']['test'], "mystring")

    def test_evaluate_configuration_int(self):
        x = self.basic_config_header +  "test: 4"
        y = core.parse_configuration_string(x)
        self.assertEqual(y['pax']['test'], 4)

    def test_evaluate_configuration_float(self):
        x = self.basic_config_header +  "test: 4.0"
        y = core.parse_configuration_string(x)
        self.assertEqual(y['pax']['test'], 4.0)

    def test_evaluate_configuration_add(self):
        x = self.basic_config_header +  "test: 4.0 + 2.0"
        y = core.parse_configuration_string(x)
        self.assertEqual(y['pax']['test'], 6.0)

    def test_evaluate_configuration_units(self):
        x = self.basic_config_header +  "test: 4.0 * mm"
        y = core.parse_configuration_string(x)
        self.assertEqual(y['pax']['test'], 4.0 * units.mm)

    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()
