#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""

import unittest
import inspect
import tempfile
import shutil
import os

from pax import core, plugin, datastructure

dummy_plugin = """
from pax import plugin
class FunkyTransform(plugin.TransformPlugin):
    gnork = True
"""


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
                                                    'encoder_plugin': None,
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

    def test_no_configuration(self):
        """ Test RuntimeError when no configuration
        """
        with self.assertRaises(RuntimeError):
            core.Processor()

    def test_custom_plugin_location(self):
        """Tests loading a plugin from a custom location"""
        tempdir = tempfile.mkdtemp()
        with open(os.path.join(tempdir, 'temp_plugin_file.py'), mode='w') as outfile:
            outfile.write(dummy_plugin)
        mypax = core.Processor(config_dict={'pax': {'plugin_group_names':   ['bla'],
                                                    'plugin_paths':         [tempdir],
                                                    'bla':                  'temp_plugin_file.FunkyTransform'}},
                               just_testing=True)
        self.assertIsInstance(mypax, core.Processor)
        pl = mypax.get_plugin_by_name('FunkyTransform')
        self.assertIsInstance(pl, plugin.TransformPlugin)
        self.assertTrue(hasattr(pl, 'gnork'))
        shutil.rmtree(tempdir)

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
        config = {'pax': {'events_to_process': [0],
                          'encoder_plugin': None,
                          'output': 'Dummy.DummyOutput'}}
        mypax = core.Processor(config_names='XENON100', config_dict=config)
        mypax.run()

    def test_process_single_xed_event_olddsp(self):
        """ Process the first event from the XED file using Xerawdp matching config
        """
        mypax = core.Processor(config_names=['XENON100', 'XerawdpImitation'],
                               config_dict={'pax': {
                                   'events_to_process': [0],
                                   'encoder_plugin': None,
                                   'output': 'Dummy.DummyOutput'}})
        mypax.run()
        pl = mypax.get_plugin_by_name('DummyOutput')
        self.assertIsInstance(pl, plugin.OutputPlugin)
        e = pl.last_event
        self.assertIsInstance(e, datastructure.Event)
        # Check that the peak areas remain the same
        self.assertEqual([x.area for x in e.peaks],
                         [176279.0866616674, 736.576200034518, 611.0961166092862,
                          129.12023409842166, 88.43269016354068, 16.19189004866498,
                          430.28177885354137, 1.9494012301013646, 1.52583418095758,
                          1.5248293965443862, 1.5214431653719354, 1.1431245035453605,
                          1.119126521198631, 0.8370846398458212, 0.3965132112382404])

    def test_process_event_list(self):
        """ Take a list of event numbers from a file
        """
        with open('temp_eventlist.txt', mode='w') as outfile:
            outfile.write("0\n7\n")
        config = {'pax': {'event_numbers_file': 'temp_eventlist.txt',
                          'plugin_group_names': ['input', 'output'],
                          'encoder_plugin': None,
                          'output': 'Dummy.DummyOutput'}}
        mypax = core.Processor(config_names='XENON100', config_dict=config)
        mypax.run()
        self.assertEqual(mypax.get_plugin_by_name('DummyOutput').last_event.event_number, 7)
        os.remove('temp_eventlist.txt')

    def test_simulator(self):
        """ Process the events in dummy_waveforms.csv
        """
        mypax = core.Processor(config_names=['XENON100', 'Simulation'],
                               config_dict={'pax': {
                                   'encoder_plugin': None,
                                   'output': 'Dummy.DummyOutput'}})
        mypax.run()
        pl = mypax.get_plugin_by_name('DummyOutput')
        self.assertIsInstance(pl, plugin.OutputPlugin)
        # TODO: do some checks on the simulator output

    def test_plotting(self):
        """ Plot the first event from the default XED file
        """
        import matplotlib
        # Force matplotlib to switch to a non-GUI backend, so the test runs on Travis
        matplotlib.pyplot.switch_backend('Agg')
        mypax = core.Processor(config_names='XENON100',
                               config_dict={'pax': {'output': 'Plotting.PlotEventSummary',
                                                    'pre_output': [],
                                                    'encoder_plugin': None,
                                                    'events_to_process': [0],
                                                    'output_name': 'plots_test'}})
        mypax.run()
        self.assertTrue(os.path.exists('./plots_test'))
        self.assertTrue(os.path.exists('./plots_test/000000_000000.png'))
        shutil.rmtree('plots_test')


if __name__ == '__main__':
    unittest.main()
