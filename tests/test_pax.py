#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""

import unittest
import inspect

from pax import core, plugin, datastructure


class TestPax(unittest.TestCase):

    def test_pax_minimal(self):
        """ The smallest possible test that actually instantiates the processor.
        Does not load any plugins or configuration
        """
        mypax = core.Processor(config_dict={'pax': {}}, just_testing=True)
        self.assertIsInstance(mypax, core.Processor)
        # Make sure the configuration is actually empty, and a default config did not sneakily get loaded...
        # Note plugin_group_names gets autoset during config init
        self.assertEqual(mypax.config, {'DEFAULT': {}, 'pax': {'plugin_group_names': []}})

    def test_pax_config_string(self):
        """ Similar, but using an almost-empty config string """
        mypax = core.Processor(config_string="[pax]", just_testing=True)
        self.assertIsInstance(mypax, core.Processor)
        # Make sure the configuration is actually empty, and a default config did not sneakily get loaded...
        # Note plugin_group_names gets autoset during config init
        self.assertEqual(mypax.config, {'DEFAULT': {}, 'pax': {'plugin_group_names': []}})

    ##
    # Test for configuration evaluation
    ##

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

    ##
    # Tests for plugin instantiation
    ##

    def test_dummy_input_plugin(self):
        mypax = core.Processor(config_dict={'pax': {'plugin_group_names':   ['input'],
                                                    'input':                'Dummy.DummyInput'}},
                               just_testing=True)
        self.assertIsInstance(mypax, core.Processor)
        self.assertIsInstance(mypax.input_plugin, plugin.InputPlugin)

    def test_dummy_output_plugin(self):
        mypax = core.Processor(config_dict={'pax': {'plugin_group_names':   ['output'],
                                                    'output':                'Dummy.DummyOutput'}},
                               just_testing=True)
        self.assertIsInstance(mypax, core.Processor)
        self.assertIsInstance(mypax.action_plugins[0], plugin.OutputPlugin)

    def test_dummy_transform_plugin(self):
        mypax = core.Processor(config_dict={'pax': {'plugin_group_names':   ['bla'],
                                                    'bla':                'Dummy.DummyTransform'}},
                               just_testing=True)
        self.assertIsInstance(mypax, core.Processor)
        self.assertIsInstance(mypax.action_plugins[0], plugin.TransformPlugin)

    def test_get_plugin_by_name(self):
        mypax = core.Processor(config_dict={'pax': {'plugin_group_names':   ['bla'],
                                                    'bla':                  ['Dummy.DummyTransform',
                                                                             'Dummy.DummyTransform2',
                                                                             'Dummy.DummyTransform3']}},
                               just_testing=True)
        self.assertIsInstance(mypax, core.Processor)
        pl = mypax.get_plugin_by_name('DummyTransform2')
        self.assertIsInstance(pl, plugin.TransformPlugin)
        self.assertEqual(pl.__class__.__name__, 'DummyTransform2')

    def test_get_input_plugin_by_name(self):
        mypax = core.Processor(config_dict={'pax': {'plugin_group_names':   ['input'],
                                                    'input':                'Dummy.DummyInput'}},
                               just_testing=True)
        pl = mypax.get_plugin_by_name('DummyInput')
        self.assertIsInstance(pl, plugin.InputPlugin)
        self.assertEqual(pl.__class__.__name__, 'DummyInput')

    def test_evaluate_default_configuration(self):
        """ Test loading the entire default configuration & all its plugins
        Will trigger a warning: no configuration specified
        """
        mypax = core.Processor()
        self.assertIsInstance(mypax, core.Processor)
        self.assertIsInstance(mypax.input_plugin,  plugin.InputPlugin)
        self.assertTrue(len(mypax.action_plugins) > 0)
        for p in mypax.action_plugins:
            self.assertIsInstance(p, (plugin.TransformPlugin, plugin.OutputPlugin))
            p.shutdown()    # To close output file we just wrote, normally happens after .run()

    ##
    # Test event processing
    ##

    def test_get_events(self):
        """ Test getting events from the input plugin
        """
        mypax = core.Processor(config_dict={'pax': {'plugin_group_names':   ['input'],
                                                    'input':                'Dummy.DummyInput'}},
                               just_testing=True)
        self.assertTrue(inspect.isgeneratorfunction(mypax.get_events))
        event_generator = mypax.get_events()
        event = next(event_generator)
        self.assertIsInstance(event, datastructure.Event)

    def test_process_empty_event(self):
        """ Test processing without processing plugins defined
        """
        mypax = core.Processor(config_dict={'pax': {'plugin_group_names':   ['input'],
                                                    'input':                'Dummy.DummyInput'}},
                               just_testing=True)
        event_generator = mypax.get_events()
        event = next(event_generator)
        event = mypax.process_event(event)
        self.assertIsInstance(event, datastructure.Event)

    def test_process_single_xed_event(self):
        """ Process the first event from the XED file.
        """
        # TODO: delete the HD5 file that is created by this
        mypax = core.Processor(config_names='XED', config_dict={'pax': {'events_to_process': [0]}})
        mypax.run()

if __name__ == '__main__':
    unittest.main()
