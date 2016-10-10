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
                                          'test':               'CheckPulses.CheckBoundsAndCount'},
                                      'CheckPulses.CheckBoundsAndCount': {
                                          'truncate_pulses_partially_outside': True}})
        self.plugin = self.pax.get_plugin_by_name('CheckBoundsAndCount')
        self.baseline = self.pax.config['DEFAULT']['digitizer_reference_baseline']

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')

    def make_single_pulse_event(self, **kwargs):
        event = datastructure.Event(
            n_channels=10,
            start_time=0,
            length=100,
            sample_duration=self.pax.config['DEFAULT']['sample_duration']
        )
        event.pulses.append(datastructure.Pulse(**kwargs))
        return event

    def test_basic_pulse(self):
        event = self.make_single_pulse_event(
            channel=1,
            left=1,
            raw_data=np.ones(20, dtype=np.int16) * self.baseline)

        self.plugin.transform_event(event)
        self.assertEqual(event.n_pulses_per_channel[0], 0)
        self.assertEqual(event.n_pulses_per_channel[1], 1)
        self.assertEqual(event.n_pulses, 1)

    def test_strange_pulses(self):
        # One sample at end
        event = self.make_single_pulse_event(
            channel=1,
            left=99,
            raw_data=np.ones(1, dtype=np.int16) * self.baseline)
        self.plugin.transform_event(event)

        # Pulse fills event
        event = self.make_single_pulse_event(
            channel=1,
            left=0,
            raw_data=np.ones(100, dtype=np.int16) * self.baseline)
        self.plugin.transform_event(event)

    def test_pulses_outside_event(self):

        # Occ starts to early
        event = self.make_single_pulse_event(
            channel=1,
            left=-5,
            raw_data=np.ones(20, dtype=np.int16) * self.baseline)
        event = self.plugin.transform_event(event)
        oc = event.pulses[0]
        self.assertEqual(oc.left, 0)
        self.assertEqual(oc.right, 20-1-5)

        # Occ starts to early
        event = self.make_single_pulse_event(
            channel=1,
            left=-1,
            raw_data=np.ones(20, dtype=np.int16) * self.baseline)
        event = self.plugin.transform_event(event)
        oc = event.pulses[0]
        self.assertEqual(oc.left, 0)
        self.assertEqual(oc.right, 20-1-1)

        # Occ overhangs
        event = self.make_single_pulse_event(
            channel=1,
            left=90,
            raw_data=np.ones(20, dtype=np.int16) * self.baseline)
        event = self.plugin.transform_event(event)
        oc = event.pulses[0]
        self.assertEqual(oc.left, 90)
        self.assertEqual(oc.right, 99)

        # Occ starts too early AND overhangs
        event = self.make_single_pulse_event(
            channel=1,
            left=-5,
            raw_data=np.ones(200, dtype=np.int16) * self.baseline)
        event = self.plugin.transform_event(event)
        oc = event.pulses[0]
        self.assertEqual(oc.left, 0)
        self.assertEqual(oc.right, 99)

        # Occ entirely outside
        event = self.make_single_pulse_event(
            channel=1,
            left=-5,
            raw_data=np.ones(2, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.PulseBeyondEventError,
                          self.plugin.transform_event, event)

        # Occ entirely outside
        event = self.make_single_pulse_event(
            channel=1,
            left=120,
            raw_data=np.ones(2, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.PulseBeyondEventError,
                          self.plugin.transform_event, event)

        # Occ entirely outside
        event = self.make_single_pulse_event(
            channel=1,
            left=100,
            raw_data=np.ones(1, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.PulseBeyondEventError,
                          self.plugin.transform_event, event)

        # Occ entirely outside
        event = self.make_single_pulse_event(
            channel=1,
            left=-1,
            raw_data=np.ones(1, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.PulseBeyondEventError,
                          self.plugin.transform_event, event)

        # Occ entirely outside
        event = self.make_single_pulse_event(
            channel=1,
            left=-10000000,
            raw_data=np.ones(1, dtype=np.int16) * self.baseline)
        self.assertRaises(exceptions.PulseBeyondEventError,
                          self.plugin.transform_event, event)


if __name__ == '__main__':
    unittest.main()
