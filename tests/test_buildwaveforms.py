import numpy as np
import unittest

from pax import core, datastructure, exceptions


class TestBuildWaveforms(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={
            'pax': {
                'plugin_group_names': ['test'],
                'test':               'BuildWaveforms.BuildWaveforms'},
            'BuildWaveforms.BuildWaveforms': {
                'truncate_occurrences_partially_outside': False}})
        self.plugin = self.pax.get_plugin_by_name('BuildWaveforms')
        self.baseline = self.pax.config['DEFAULT']['digitizer_baseline']

    def make_single_occurrence_event(self, **kwargs):
        event = datastructure.Event(
            n_channels=10,
            start_time=0,
            length=100,
            sample_duration=self.pax.config['DEFAULT']['sample_duration']
        )
        event.occurrences.append(datastructure.Occurrence(**kwargs))
        return event

    def test_basic_occurrence(self):

        event = self.make_single_occurrence_event(
            channel=1,
            left=1,
            raw_data=np.ones(20) * self.baseline)
        event = self.plugin.transform_event(event)
        self.assertTrue(len(event.sum_waveforms) > 0)

    def test_strange_occurrences(self):
        # One sample at end
        event = self.make_single_occurrence_event(
            channel=1,
            left=99,
            raw_data=np.ones(1) * self.baseline)
        event = self.plugin.transform_event(event)
        self.assertTrue(len(event.sum_waveforms) > 0)

        # Occurrence fills event
        event = self.make_single_occurrence_event(
            channel=1,
            left=0,
            raw_data=np.ones(100) * self.baseline)
        event = self.plugin.transform_event(event)
        self.assertTrue(len(event.sum_waveforms) > 0)

    def test_occurrences_outside_event(self):
        # TODO: Also add tests for truncation itself, in case it is allowed in the config

        event = self.make_single_occurrence_event(
            channel=1,
            left=-5,
            raw_data=np.ones(20) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)

        event = self.make_single_occurrence_event(
            channel=1,
            left=-5,
            raw_data=np.ones(2) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)

        event = self.make_single_occurrence_event(
            channel=1,
            left=120,
            raw_data=np.ones(2) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)

        event = self.make_single_occurrence_event(
            channel=1,
            left=-5,
            raw_data=np.ones(200) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)

        event = self.make_single_occurrence_event(
            channel=1,
            left=100,
            raw_data=np.ones(1) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)
