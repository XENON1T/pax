from __future__ import division
import unittest

import numpy as np


from pax import units, trigger, configuration
from pax.datastructure import TriggerSignal


class TestTrigger(unittest.TestCase):

    def test_find_signals(self):

        numba_signals_buffer = np.zeros(100, dtype=TriggerSignal.get_dtype())

        def get_signals(times, separation):
            times = np.array(times, np.int64)
            n_found = trigger.find_signals(times, separation, numba_signals_buffer)
            return numba_signals_buffer[:n_found]

        # No times = no signals
        self.assertEqual(len(get_signals([], 1)), 0)

        # No coincidence = no signals
        self.assertEqual(len(get_signals([1], 1)), 0)
        self.assertEqual(len(get_signals([1, 10], 1)), 0)

        # Simple coincidence = one signal
        self.assertEqual(len(get_signals([1, 1], 1)), 1)

        # Test for off-by-one error
        self.assertEqual(len(get_signals([1, 3], 2)), 0)
        self.assertEqual(len(get_signals([1, 2], 2)), 1)

        # Test left and right time
        self.assertEqual(get_signals([1, 2], 2)['left_time'], 1)
        self.assertEqual(get_signals([1, 2], 2)['right_time'], 2)

        # Test multiple signals and signal properties
        sigs = get_signals([1, 2, 100, 101, 102], 2)
        self.assertEqual(len(sigs), 2)
        self.assertEqual(sigs[0]['left_time'], 1)
        self.assertEqual(sigs[0]['right_time'], 2)
        self.assertEqual(sigs[0]['n_pulses'], 2)
        self.assertAlmostEqual(sigs[0]['time_mean'], np.mean([1, 2]))
        self.assertAlmostEqual(sigs[0]['time_rms'], np.std([1, 2]))
        self.assertEqual(sigs[0]['left_time'], 1)
        self.assertEqual(sigs[1]['left_time'], 100)
        self.assertEqual(sigs[1]['right_time'], 102)
        self.assertEqual(sigs[1]['n_pulses'], 3)
        self.assertAlmostEqual(sigs[1]['time_mean'], np.mean([100, 101, 102]))
        self.assertAlmostEqual(sigs[1]['time_rms'], np.std([100, 101, 102]))

    def test_find_event_ranges(self):
        config = configuration.load_configuration('XENON1T')['Trigger']
        config['event_split_threshold'] = 3
        config['left_extension'] = 0
        config['right_extension'] = 0
        trig = trigger.Trigger(config)

        a = np.array([0, 0, 1, 4, 5, 10])
        np.testing.assert_array_equal(trig.find_event_ranges(a),
                                      np.array([[0, 1], [4, 5], [10, 10]], dtype=np.int))

    def test_group_signals(self):
        example_signals = np.zeros(10, dtype=TriggerSignal.get_dtype())
        example_signals['left_time'] = np.arange(10)
        buffer = np.zeros((2, 2), dtype=np.int)

        trigger.group_signals(example_signals, np.array([[2, 4], [8, 50]], dtype=np.int), buffer)

        self.assertEqual(buffer.tolist(), [[2, 5], [8, 9]])

    def test_trigger(self):
        """Integration test for the trigger"""
        # Configure a trigger to always trigger on any signal,
        # and not have any left and right extension (for simplicity)
        config = configuration.load_configuration('XENON1T')['Trigger']
        config['trigger_probability'] = {0: {2: 1},
                                         1: {2: 1},
                                         2: {2: 1}}
        config['left_extension'] = 0
        config['right_extension'] = 0

        offset_per_block = int(10 * units.ms)

        # A fake data generator.
        # Each list in `data` below is passed in turn to the trigger as a numpy array,
        # with an offset of offset_per_block * block i
        def data_maker():
            data = [
                # 0: No pulses (just to check no crash)
                [],
                # 1: No signal = no trigger
                [0, 10 * units.us],
                # 2: Trigger (S1) at 0 ns
                [0, 1],
                # 3: Trigger (S2) at 0 ns
                np.arange(0, 1 * units.us, 0.1 * units.us),
                # 4: Two events (S1, then S1 way later) at 0ns and 3000000 ns
                [0, 1] + [3 * units.ms, 3 * units.ms + 1],
                # 5: One event with two triggers, 0 - 300000 ns  (note one less zero after 3 than above)
                [0, 1] + [0.3 * units.ms, 0.3 * units.ms + 1],
                # 6: No trigger yet, wait for more data
                [offset_per_block],
                # 7: Trigger at 0 ns
                [0],
            ]
            for i, d in enumerate(data):
                d = np.array(d, dtype=np.int)
                d += i * offset_per_block
                yield d, d.max() if len(d) else 0

        event_ranges = []
        should_get = [[20000000, 20000000],
                      [30000000, 30000000],
                      [40000000, 40000000],
                      [43000000, 43000000],
                      [50000000, 50300000],
                      [70000000, 70000000]]

        data_gen = data_maker()
        trig = trigger.Trigger(config)

        while trig.more_data_is_coming:
            try:
                x, last_time_searched = next(data_gen)
                trig.add_new_data(x, last_time_searched)
            except StopIteration:
                trig.more_data_is_coming = False

            for event_range, signals in trig.get_trigger_ranges():
                # Trigger gave us a new event range: push it to the queue
                event_ranges.append(event_range.tolist())

        for i in range(len(should_get)):
            self.assertEqual(should_get[i], event_ranges[i])

        self.assertEqual(event_ranges, should_get)


if __name__ == '__main__':
    import sys
    import logging
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("Trigger").setLevel(logging.DEBUG)
    unittest.main()
