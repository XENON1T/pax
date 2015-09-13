import unittest

from pax import core, datastructure


class TestCheckPulses(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100',
                                  just_testing=True,
                                  config_dict={
                                      'pax': {
                                          'plugin_group_names': ['test'],
                                          'test':               'CheckPulses.ConcatenateAdjacentPulses'}})
        self.plugin = self.pax.get_plugin_by_name('ConcatenateAdjacentPulses')

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')

    def test_concatenation(self):
        for pulse_bounds, concatenated_pulse_bounds in (
            ([[0, 1], [4, 5]],                   [[0, 1], [4, 5]]),
            ([[0, 1], [2, 5]],                   [[0, 5]]),
            ([[0, 0], [1, 5]],                   [[0, 5]]),
            ([[0, 0], [1, 2], [3, 3], [4, 5]],   [[0, 5]]),
            ([[0, 0], [1, 2], [3, 5]],           [[0, 5]]),
            ([[0, 0], [1, 1], [3, 5]],           [[0, 1], [3, 5]]),
            ([], []),
        ):
            e = datastructure.Event(n_channels=self.plugin.config['n_channels'],
                                    start_time=0,
                                    sample_duration=self.plugin.config['sample_duration'],
                                    stop_time=int(1e6),
                                    pulses=[dict(left=l, right=r, channel=1) for l, r in pulse_bounds])
            e = self.plugin.transform_event(e)
            found_pulse_bounds = [[p.left, p.right] for p in e.pulses]
            self.assertEqual(concatenated_pulse_bounds, found_pulse_bounds)


if __name__ == '__main__':
    unittest.main()
