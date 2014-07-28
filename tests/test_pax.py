#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""

import unittest

from pax import units
from pax import pax


class TestPax(unittest.TestCase):
	def setUp(self):
		pass

	def test_processor_type(self):
		with self.assertRaises(AssertionError):
			pax.processor(4, 4, 4)

	def test_evaluate_configuration_string(self):
		x = {'test': "\"mystring\""}
		y = pax.evaluate_configuration(x)
		self.assertEqual(y['test'], "mystring")

	def test_evaluate_configuration_int(self):
		x = {'test': "4"}
		y = pax.evaluate_configuration(x)
		self.assertEqual(y['test'], 4)

	def test_evaluate_configuration_float(self):
		x = {'test': "4.0"}
		y = pax.evaluate_configuration(x)
		self.assertEqual(y['test'], 4.0)

	def test_evaluate_configuration_add(self):
		x = {'test': "4.0 + 2.0"}
		y = pax.evaluate_configuration(x)
		self.assertEqual(y['test'], 6.0)

	def test_evaluate_configuration_units(self):
		x = {'test': "4.0 * mm"}
		y = pax.evaluate_configuration(x)
		self.assertEqual(y['test'], 4.0 * units.mm)

	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest.main()
