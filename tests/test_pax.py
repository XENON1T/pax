#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""

import unittest

from pax import pax


class TestPax(unittest.TestCase):

    def setUp(self):
        pass

    def test_processor_type(self):
        with self.assertRaises(AssertionError):
            pax.Processor(4, 4, 4)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
