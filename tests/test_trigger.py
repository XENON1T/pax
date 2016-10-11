from __future__ import division
import unittest

import numpy as np
import six

from pax import units, trigger, configuration
from pax.datastructure import TriggerSignal
from pax.trigger_plugins.FindSignals import signal_finder
from pax.trigger_plugins.SaveSignals import group_signals
from pax.trigger_plugins.DeadTimeTally import DeadTimeTally
from pax.exceptions import TriggerGroupSignals
import tempfile


class TestSignalFinder(unittest.TestCase):

    def test_find_signals(self):
        numba_signals_buffer = np.zeros(100, dtype=TriggerSignal.get_dtype())

        def get_signals(start_times, separation):
            times = np.zeros(len(start_times), dtype=trigger.pulse_dtype)
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

        self.assertEqual(len(should_get), len(event_ranges))
        for i in range(len(should_get)):
            self.assertEqual(should_get[i], event_ranges[i])

        self.assertEqual(event_ranges, should_get)


class TestDeadTimeCalculation(unittest.TestCase):

    def run_test(self, on_times, off_times, return_type='summed',
                 batch_duration=np.pi * units.s,
                 end_of_run_offset=1 * units.s):
        pax_config = configuration.load_configuration('XENON1T')
        trig = trigger.Trigger(pax_config)

        # The results of the dead time tally would normally only be written to the db, not stored in the trigger
        # data datastructure.
        # So... we monkey-patch the trigger to intercept the call to save_monitor_data
        captured_data = []

        def hacked_save_monitor_data(data_type, data, metadata=None):
            # Store captured data in a function attribute, would have liked to use a variable in outer scope,
            # but we don't have nonlocal in py2...
            hacked_save_monitor_data.captured_data.append((data_type, data, metadata))
        hacked_save_monitor_data.captured_data = []

        trig.save_monitor_data = hacked_save_monitor_data
        tp = DeadTimeTally(trig, dict(dark_rate_save_interval=1*units.s))

        n_on = len(on_times)
        n_off = len(off_times)
        pulses = np.zeros(n_on + n_off, dtype=trigger.pulse_dtype)
        pulses['pmt'][:n_on] = pax_config['DEFAULT']['channels_in_detector']['busy_on'][0]
        pulses['time'][:n_on] = on_times
        pulses['pmt'][n_on:] = pax_config['DEFAULT']['channels_in_detector']['busy_off'][0]
        pulses['time'][n_on:] = off_times
        pulses.sort(order='time')

        end_of_run = max(pulses['time']) + end_of_run_offset

        # Feed the data in in batches
        batch_starts = np.arange(0, end_of_run, batch_duration)
        for batch_i, batch_start in enumerate(batch_starts):
            data = trigger.TriggerData()
            if batch_i == len(batch_starts) - 1:
                data.last_data = True
                data.last_time_searched = end_of_run
            else:
                data.last_time_searched = batch_start + end_of_run_offset
            data.pulses = pulses[(pulses['time'] >= batch_start) &
                                 (pulses['time'] < batch_start + batch_duration)]
            tp.process(data)

        captured_data = hacked_save_monitor_data.captured_data
        if return_type == 'full':
            return captured_data

        self.assertEqual(len(captured_data), np.ceil(data.last_time_searched / units.s))

        # Sum the batch data
        dead_time = 0
        for x in captured_data:
            dead_time += x[1]['busy']
        return dead_time

    def test_dead_time_calculation(self):
        # nonlocal in dirty monkeypatch below not supported in py2
        if six.PY2:
            return

        # Very basic test, some more checks
        captured_data = self.run_test([0], [10], return_type='full')
        self.assertEqual(len(captured_data), 2)
        self.assertEqual(captured_data[0][0], 'dead_time_info')
        self.assertEqual(captured_data[0][1]['time'], 1 * units.s)
        self.assertEqual(captured_data[0][1]['busy'], 10)
        self.assertEqual(captured_data[1][1]['busy'], 0)

        # Test summing data
        self.assertEqual(self.run_test([0], [10]), 10)

        # Test abrupt end
        self.assertEqual(self.run_test([0], [10], end_of_run_offset=0.2 * units.s), 10)

        # Test two intervals
        self.assertEqual(self.run_test([0, 100], [10, 110]), 20)

        # Test two intervals, spread over two batches
        self.assertEqual(self.run_test([0, units.s], [10, units.s + 10]), 20)

        # Test an interval straddling a batch
        self.assertEqual(self.run_test([units.s - 100], [units.s + 100]), 200)

        # Test a very large interval
        self.assertEqual(self.run_test([0], [13.5 * units.s]), 13.5 * units.s)


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
