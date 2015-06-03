import unittest

import numpy as np

from pax.datastructure import Event, Pulse
from pax import core


class TestZLE(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100',
                                  just_testing=True,
                                  config_dict={
                                      'pax': {
                                          'plugin_group_names': ['test'],
                                          'test':               'ZLE.SoftwareZLE'},
                                      'ZLE.SoftwareZLE': {
                                          'zle_threshold': 40,
                                          'samples_to_store_before': 50,
                                          'samples_to_store_after': 50,
                                          'max_intervals': 32,
                                      }})
        self.plugin = self.pax.get_plugin_by_name('SoftwareZLE')

    def test_zle(self):
        for w, pulse_bounds_should_be in (
            ([60],                                             [[0, 0]]),
            ([60, 60],                                         [[0, 1]]),
            ([0, 60, 60, 0],                                   [[0, 3]]),
            ([1] * 100 + [60] + [2] * 100,                     [[50, 150]]),
            ([1] * 100 + [30] + [2] * 100,                     []),
            ([1] * 100 + [60] + [2] * 200 + [60] + [3] * 100,  [[50, 150], [251, 351]]),
            ([1] * 100 + [60] + [2] * 70 + [60] + [3] * 100,   [[50, 100 + 1 + 70 + 1 + 50 - 1]]),
        ):
            w = np.array(w).astype(np.int16)
            # Convert from ADC above baseline (easier to specify) to raw ADC counts (what the plugin needs)
            w = self.plugin.config['digitizer_reference_baseline'] - w
            e = Event(n_channels=self.plugin.config['n_channels'],
                      start_time=0,
                      stop_time=int(1e6),
                      sample_duration=self.pax.config['DEFAULT']['sample_duration'],
                      pulses=[Pulse(left=0,
                                    channel=1,
                                    raw_data=w)])
            e = self.plugin.transform_event(e)
            pulse_bounds = [[pulse.left, pulse.right] for pulse in e.pulses]

            # Check the pulse bounds
            self.assertEqual(pulse_bounds, pulse_bounds_should_be)

            # Check if the data was put in correctly
            for i, (l, r) in enumerate(pulse_bounds):
                self.assertEqual(e.pulses[i].raw_data.tolist(), w[l:r + 1].tolist())

if __name__ == '__main__':
    unittest.main()
