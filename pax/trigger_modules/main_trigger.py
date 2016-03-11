"""Main trigger for XENON1T

We can (if desired) save TriggerSignals outside the events, or a subset of them, to the trigger_data hdf5 file.
This is currently an experimental feature, use with caution as it can slow the trigger down a lot.
Note the full raw data is not saved for these signals, just the TriggerSignal information (see pax.datastructure).
"""
import random
import numba
import numpy as np

from pax import units, trigger
from pax.trigger_modules.base import TriggerModule


class MainTrigger(TriggerModule):
    numeric_id = 0
    saved_signals = []

    def startup(self):
        config = self.config

        # Validate the configuration
        # If no left and right extension specified, set them to half the split gap (floored) minus 1
        if config.get('left_extension', None) is None:
            self.log.warning("No left/right extensions specified: using half of event separation")
            config['left_extension'] = config['right_extension'] = np.floor(config['event_separation'] / 2) - 1
        if config['event_separation'] <= config['left_extension'] + config['right_extension']:
            raise ValueError("event_separation must be larger than left + right extension!")
        config['left_extension'] = abs(config['left_extension'])            # Deal with Chris' shenanigans

        self.log.info("Starting XENON1T main trigger")
        self.log.info("\tEvent separation threshold: %0.2f ms", config['event_separation'] / units.ms)
        self.log.info("\tLeft extension: %0.2f us", config['left_extension'] / units.us)
        self.log.info("\tRight extension: %0.2f us", config['right_extension'] / units.us)

        # Convert trigger_probability dictionary of dictionaries to 2d array, with intermediate values filled in
        self.log.info("\tTrigger probabilities: %s" % str(config['trigger_probability']))
        p_length = max([max(ps.keys()) for ps in config['trigger_probability'].values()]) + 1
        p_matrix = np.zeros((3, p_length), dtype=np.float)
        for sig_type, ps in config['trigger_probability'].items():
            for i, n in enumerate(sorted(ps.keys())):
                p_matrix[sig_type][n:] = ps[n]
        self.p_matrix = p_matrix

        if self.config['save_signals_outside_events']:
            self.outside_signals_dataset = self.trigger.f.create_dataset('signals_outside_events',
                                                                         shape=(0,),
                                                                         maxshape=(None,),
                                                                         dtype=self.trigger.numba_signals_buffer.dtype,
                                                                         compression="gzip")

    def get_event_ranges(self, signals):
        self.log.debug("Main trigger received %d signals" % len(signals))

        # Classify the signals; modifies signals in-place.
        classify_signals(signals,
                         s1_max_rms=self.config['s1_max_rms'],
                         s2_min_pulses=self.config['s2_min_pulses'])

        # Determine which signals trigger an event; modifies signals in-place.
        decide_triggers(signals, p_matrix=self.p_matrix)

        # Are there any signals we saved from last time? If so, prepend them now, before the final grouping into events.
        # We can do this after classification and the trigger decision, since these are purely per-signal operations.
        # The concatenate will copy signals to a new array, so if we did this earlier, future triggers
        # would not be able to use the classification information we've provided here.
        if len(self.saved_signals):
            self.log.debug("%d saved signals from last time left" % len(self.saved_signals))
            signals = np.concatenate((self.saved_signals, signals))

        # Group the triggers into event ranges
        trigger_times = signals[signals['trigger']]['left_time']
        event_ranges = self.find_event_ranges(trigger_times)

        if self.trigger.more_data_coming and len(signals):
            # The interpretation of the final few signals signals could change after new data comes in.
            # Save all such signals for later, we'll look at them again once that data arrives.
            # If any event is in this 'unsafe range', do not send it; it will be re-made the next time.
            self.lookback_time = self.trigger.last_time_searched - self.config['event_separation']

            if len(event_ranges) and event_ranges[-1][-1] > self.lookback_time:
                # The last event range can still be extended by future data: we should not send it yet.
                # Instead, save the signals that are part of it, and look at them again next time.
                self.lookback_time = min(self.lookback_time, event_ranges[-1][0] - self.config['left_extension'])
                event_ranges = event_ranges[:-1]
                self.log.debug("Not sending last event yet, will be remade next time")

            # What is the last signal we should revisit later?
            # TODO: Can be sped up by loop in numba. But probably doesn't matter much, will only run a few iterations.
            i = 0
            for i, t in enumerate(signals['left_time'][::-1]):
                if t < self.lookback_time:
                    break
            last_index_before_lookback = len(signals) - 1 - i
            if last_index_before_lookback == len(signals) - 1:
                self.log.debug("Keeping %d signals after %d in buffer until next data comes" % (
                    len(signals) - 1 - last_index_before_lookback, self.lookback_time))
            else:
                self.log.debug("All signals are safe to consider in event building")
            self.saved_signals = signals[last_index_before_lookback + 1:]

        yield from self.make_events(event_ranges, signals)

        # Save basic info about signals outside events to the trigger_data hdf5, if desired.
        # TODO: if needed, this can probably be sped up a lot by using a numba routine
        if self.config['save_signals_outside_events']:
            signal_is_in_event = np.zeros(len(signals), dtype=np.bool)
            for event_i in range(len(event_ranges)):
                signal_start_i, signal_end_i = self.signal_indices_buffer[event_i]
                signal_is_in_event[signal_start_i:signal_end_i + 1] = True

            outsigs = signals[(True ^ signal_is_in_event)]
            outsigs = np.concatenate([outsigs[(outsigs['type'] == sigtype) &
                                              (outsigs['n_pulses'] >=
                                               self.config['outside_signals_save_thresholds'][sigtype])]
                                      for sigtype in range(2 + 1)])

            if len(outsigs):
                self.log.debug("Storing %d signals outside events" % len(outsigs))
                trigger.h5py_append(self.outside_signals_dataset, outsigs)

    def find_event_ranges(self, trigger_times):
        """Return event ranges corresponding to trigger_times,
        grouping nearby triggers and applying left and right extension"""
        event_ranges = []
        left_ext = self.config['left_extension']
        right_ext = self.config['right_extension']
        max_l = self.config['max_event_length']
        for group_of_tr in split_on_gap(trigger_times, self.config['event_separation']):
            start = group_of_tr[0] - left_ext
            stop = group_of_tr[-1] + right_ext
            if stop - start > max_l:
                self.log.warning("Event %d-%d too long (%0.2f ms), truncated to %0.2f ms. "
                                 "Consider changing trigger settings!" % (start, stop,
                                                                          (stop - start) / units.ms,
                                                                          max_l / units.ms))
            event_ranges.append((start, stop))
        return np.array(event_ranges, dtype=np.int)


@numba.jit(nopython=True)
def classify_signals(signals, s1_max_rms, s2_min_pulses):
    """Set the type field of signals to 0 (unknown), 1 (s1) or 2 (s2). Modifies signals in-place.
    """
    for signal_i, s in enumerate(signals):
        sigtype = 0
        if s.time_rms > s1_max_rms:
            if s.n_pulses >= s2_min_pulses:
                sigtype = 2
        else:
            sigtype = 1
        signals[signal_i].type = sigtype


@numba.jit(nopython=True)
def decide_triggers(signals, p_matrix):
    """Decide which signals trigger, modifying signals in-place.
    p_matrix[signal_type][n_pulses] is the probability of a signal of type signal_type and n_pulses pulses to trigger
    The last entry of each signal type is used for signals with more pulses than the matrix is wide.
    """
    largest_n_in_p = len(p_matrix[0]) - 1
    for i, s in enumerate(signals):
        p = p_matrix[s.type][min(largest_n_in_p, s.n_pulses)]

        if p == 1.0:
            s.trigger = True
        elif p == 0.0:
            s.trigger = False
        else:
            s.trigger = random.random() < p


def split_on_gap(a, threshold):
    """Split a into list of arrays, each separated by threshold or more"""
    if not len(a):
        return []
    if len(a) < 2:
        return [a]
    return np.split(a, np.where(np.diff(a) >= threshold)[0] + 1)
