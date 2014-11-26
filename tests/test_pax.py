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
        pass

    def _do_config_eval_test(self, test_value, must_become):
        self.assertEqual(
            core.Processor(config_string="[pax]\ntest: %s" % test_value, just_testing=True).config['pax']['test'],
            must_become)

    def test_evaluate_configuration_string(self):
        self._do_config_eval_test('"mystring"', 'mystring')

    def test_evaluate_configuration_int(self):
        self._do_config_eval_test('4', 4)

    def test_evaluate_configuration_float(self):
        self._do_config_eval_test('4.0', 4.0)

    def test_evaluate_configuration_add(self):
        self._do_config_eval_test('4.0 + 2.0', 6.0)

    def test_evaluate_configuration_add(self):
        self._do_config_eval_test('4.0 * mm', 4.0 * units.mm)

    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()
