import unittest

import numpy as np

from pax import core, datastructure, exceptions


class TestCheckPulses(unittest.TestCase):

    def setUp(self):  # noqa
        self.pax = core.Processor(config_names='XENON100',
                                  just_testing=True,
                                  config_dict={
                                      'pax': {
                                          'plugin_group_names': ['test'],
                                          'test':               'CheckPulses.CheckBounds'},
                                      'CheckPulses.CheckBounds': {
                                          'truncate_occurrences_partially_outside': True}})
        self.plugin = self.pax.get_plugin_by_name('CheckBounds')
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
            raw_data=np.ones(20, dtype=np.int16) * self.baseline)
        self.plugin.transform_event(event)

    def test_strange_occurrences(self):
        # One sample at end
        event = self.make_single_occurrence_event(
            channel=1,
            left=99,
            raw_data=np.ones(1, dtype=np.int16) * self.baseline)
        self.plugin.transform_event(event)

        # Occurrence fills event
        event = self.make_single_occurrence_event(
            channel=1,
            left=0,
            raw_data=np.ones(100, dtype=np.int16) * self.baseline)
        self.plugin.transform_event(event)

    def test_occurrences_outside_event(self):

        # Occ starts to early
        event = self.make_single_occurrence_event(
            channel=1,
            left=-5,
            raw_data=np.ones(20, dtype=np.int16) * self.baseline)
        event = self.plugin.transform_event(event)
        oc = event.occurrences[0]
        self.assertEqual(oc.left, 0)
        self.assertEqual(oc.right, 20-1-5)

        # Occ starts to early
        event = self.make_single_occurrence_event(
            channel=1,
            left=-1,
            raw_data=np.ones(20, dtype=np.int16) * self.baseline)
        event = self.plugin.transform_event(event)
        oc = event.occurrences[0]
        self.assertEqual(oc.left, 0)
        self.assertEqual(oc.right, 20-1-1)

        # Occ overhangs
        event = self.make_single_occurrence_event(
            channel=1,
            left=90,
            raw_data=np.ones(20, dtype=np.int16) * self.baseline)
        event = self.plugin.transform_event(event)
        oc = event.occurrences[0]
        self.assertEqual(oc.left, 90)
        self.assertEqual(oc.right, 99)

        # Occ starts too early AND overhangs
        event = self.make_single_occurrence_event(
            channel=1,
            left=-5,
            raw_data=np.ones(200, dtype=np.int16) * self.baseline)
        event = self.plugin.transform_event(event)
        oc = event.occurrences[0]
        self.assertEqual(oc.left, 0)
        self.assertEqual(oc.right, 99)

        # Occ entirely outside
        event = self.make_single_occurrence_event(
            channel=1,
            left=-5,
            raw_data=np.ones(2, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)

        # Occ entirely outside
        event = self.make_single_occurrence_event(
            channel=1,
            left=120,
            raw_data=np.ones(2, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)

        # Occ entirely outside
        event = self.make_single_occurrence_event(
            channel=1,
            left=100,
            raw_data=np.ones(1, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)

        # Occ entirely outside
        event = self.make_single_occurrence_event(
            channel=1,
            left=-1,
            raw_data=np.ones(1, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.OccurrenceBeyondEventError,
                          self.plugin.transform_event, event)


if __name__ == '__main__':
    unittest.main()
