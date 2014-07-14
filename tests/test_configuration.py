#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax_configuration
----------------------------------

Tests for `pax.configuration` module.
"""

import unittest

from confiture import Confiture
from pax.configuration import PaxSchema


class TestPaxConfiguration(unittest.TestCase):

    def setUp(self):
        pass

    def test_parsing(self):
        schema = PaxSchema()
        config = Confiture("", schema=schema)
        pconfig = config.parse()
        pconfig.to_dict()

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
