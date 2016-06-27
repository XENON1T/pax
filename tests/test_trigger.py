from __future__ import division
import unittest

import numpy as np

from pax import units, trigger, configuration
from pax.datastructure import TriggerSignal
from pax.trigger_plugins.FindSignals import signal_finder
from pax.trigger_plugins.SaveSignals import group_signals
from pax.exceptions import TriggerGroupSignals
import tempfile


class TestSignalFinder(unittest.TestCase):

    def test_find_signals(self):
        numba_signals_buffer = np.zeros(100, dtype=TriggerSignal.get_dtype())

        def get_signals(start_times, separation):
            times = np.zeros(len(start_times), dtype=trigger.times_dtype)
            times['time'] = start_times
            times['area'] = 10

            sigf = signal_finder(times=times,
                                 signal_separation=separation,
                                 signal_buffer=numba_signals_buffer,

                                 next_save_time=int(10 * units.s),
                                 dark_rate_save_interval=int(10 * units.s),

                                 all_pulses_tally=np.zeros(1),
                                 lone_pulses_tally=np.zeros(1),
                                 coincidence_tally=np.zeros((1, 1)),

                                 gain_conversion_factors=10 * np.ones(1, dtype=np.float64))
            n_found = next(sigf)
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
        self.assertEqual(sigs[0]['area'], 200)
        self.assertAlmostEqual(sigs[0]['time_mean'], np.mean([1, 2]))
        self.assertAlmostEqual(sigs[0]['time_rms'], np.std([1, 2]))
        self.assertEqual(sigs[0]['left_time'], 1)
        self.assertEqual(sigs[1]['left_time'], 100)
        self.assertEqual(sigs[1]['right_time'], 102)
        self.assertEqual(sigs[1]['n_pulses'], 3)
        self.assertEqual(sigs[1]['area'], 300)
        self.assertAlmostEqual(sigs[1]['time_mean'], np.mean([100, 101, 102]))
        self.assertAlmostEqual(sigs[1]['time_rms'], np.std([100, 101, 102]))


class TestSaveSignals(unittest.TestCase):

    def test_save_signals(self):
        """Tests the logic which figures out which signals belong with which event ranges."""

        def do_test(sig_times, event_ranges):
            example_signals = np.zeros(len(sig_times), dtype=TriggerSignal.get_dtype())
            example_signals['left_time'] = sig_times

            sigind_buffer = np.zeros((100, 2), dtype=np.int)
            is_in_event = np.zeros(len(example_signals), dtype=np.bool)
            group_signals(signals=example_signals,
                          event_ranges=np.array(event_ranges, dtype=np.int),
                          signal_indices_buffer=sigind_buffer,
                          is_in_event=is_in_event)

            return sigind_buffer, is_in_event

        sigind_buffer, is_in_event = do_test(sig_times=np.arange(10), event_ranges=[[2, 4], [8, 50]])
        self.assertEqual(sigind_buffer.tolist()[:3], [[2, 4], [8, 9], [0, 0]])
        self.assertEqual(is_in_event.tolist(), [False, False, True, True, True, False, False, False, True, True])

        # Test adjacent events
        sigind_buffer, is_in_event = do_test(sig_times=np.arange(10), event_ranges=[[0, 3], [4, 5]])
        self.assertEqual(sigind_buffer.tolist()[:3], [[0, 3], [4, 5], [0, 0]])
        self.assertEqual(is_in_event.tolist(), [True, True, True, True, True, True, False, False, False, False])

        # Test single event
        sigind_buffer, is_in_event = do_test(sig_times=np.arange(10), event_ranges=[[-5, 2]])
        self.assertEqual(sigind_buffer.tolist()[:3], [[0, 2], [0, 0], [0, 0]])
        self.assertEqual(is_in_event.tolist(), [True, True, True, False, False, False, False, False, False, False])

        # Test error on event without signals
        self.assertRaises(TriggerGroupSignals, do_test, sig_times=[0, 1, 10, 11], event_ranges=[[2, 3]])


class TestGroupTriggers(unittest.TestCase):

    def test_group_trigges(self):
        """Tests the grouping of triggers into event ranges"""
        trig_times = [0, 0, 1, 4, 5, 10]

        from pax.trigger_plugins.GroupTriggers import GroupTriggers
        trig = trigger.Trigger(configuration.load_configuration('XENON1T'))
        tp = GroupTriggers(trig, dict(event_separation=3,
                                      max_event_length=10000,
                                      left_extension=0,
                                      right_extension=0))
        data = trigger.TriggerData()
        data.signals = np.zeros(len(trig_times), dtype=TriggerSignal.get_dtype())
        data.signals['trigger'] = True
        data.signals['left_time'] = trig_times
        tp.process(data)

        np.testing.assert_array_equal(data.event_ranges,
                                      np.array([[0, 1], [4, 5], [10, 10]], dtype=np.int))


class TestTriggerIntegration(unittest.TestCase):
    """Integration test for the trigger"""

    def test_trigger(self):
        # Configure a trigger to always trigger on any signal,
        # and not have any left and right extension (for simplicity)
        config = configuration.load_configuration('XENON1T')
        tempf = tempfile.NamedTemporaryFile()
        config['Trigger'].update(dict(signal_separation=1 * units.us,
                                      event_separation=1 * units.ms,
                                      left_extension=0,
                                      right_extension=0,
                                      trigger_data_filename=tempf.name))
        config['Trigger.FindSignals']['numba_signal_buffer_size'] = 1    # So we test buffer exhausted logic too
        config['Trigger.DecideTriggers'].update(dict(trigger_probability={0: {2: 1},
                                                                          1: {2: 1},
                                                                          2: {2: 1}}))
        config['Trigger.ClassifySignals'].update(dict(s1_max_rms=30 * units.ns,
                                                      s2_min_pulses=7))
        config['Trigger.GroupTriggers']['max_event_length'] = 10 * units.ms
        config['Trigger.SaveSignals']['save_signals_outside_events'] = False

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
                print("\n\n>>> %d <<<\n\n" % i)
                d = np.array(d, dtype=np.int)
                d += i * offset_per_block
                yield d, (d.max() if len(d) else 0), i == len(data) - 1

        event_ranges = []
        should_get = [(20000000, 20000000),
                      (30000000, 30000000),
                      (40000000, 40000000),
                      (43000000, 43000000),
                      (50000000, 50300000),
                      (70000000, 70000000)]

        trig = trigger.Trigger(config)

        for start_times, last_time_searched, is_last in data_maker():
            for event_range, signals in trig.run(last_time_searched=last_time_searched,
                                                 start_times=start_times,
                                                 last_data=is_last):
                print("Received event range ", event_range)
                event_ranges.append(event_range)

        trig.shutdown()

        print("Trigger finished. Event ranges received: %s" % event_ranges)

        for i in range(len(should_get)):
            self.assertEqual(should_get[i], event_ranges[i])

        self.assertEqual(event_ranges, should_get)


if __name__ == '__main__':
    import sys
    import logging
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("Trigger").setLevel(logging.DEBUG)
    # Make sure the trigger plugins' loggers are set to debug loglevel
    config = configuration.load_configuration('XENON1T')
    for pname in config['Trigger']['trigger_plugins']:
        logging.getLogger(pname).setLevel(logging.DEBUG)
    unittest.main()
