#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""
import unittest

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecWeightedSum(unittest.TestCase):

    def setUp(self):  # noqa
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={'pax': {
            'plugin_group_names': ['test'],
            'test':               'WeightedSum.PosRecWeightedSum'}})
        self.posrec_plugin = self.pax.get_plugin_by_name('PosRecWeightedSum')

        self.e = Event.empty_event()

        self.e.peaks.append(Peak({'left':  5,
                                  'right': 9,
                                  'type':  's2'}))

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'posrec_plugin')
        delattr(self, 'e')

    def test_something(self):
        self.assertIsInstance(self.posrec_plugin, plugin.TransformPlugin)
        self.assertEqual(self.posrec_plugin.__class__.__name__, 'PosRecWeightedSum')


if __name__ == '__main__':
    unittest.main()
