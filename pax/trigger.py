"""XENON1T trigger
"""
import time
import logging
import random

import numba
import numpy as np

import pax          # For version number
from pax import units
from pax.datastructure import TriggerSignal


class Trigger(object):
    more_data_is_coming = True

    # Buffer for data taken so far. Will be extended as needed.
    times = np.zeros(0, dtype=np.int64)
    last_time_searched = 0

    # Statistics for dumping at end of run
    events_built = 0
    total_event_length = 0
    pulses_read = 0

    def __init__(self, config):
        self.log = log = logging.getLogger('Trigger')

        # Validate the configuration
        # If no left and right extension specified, set them to floor(half the split gap)
        if config.get('left_extension', None) is None:
            log.warning("No left/right extensions specified: using half of event separation")
            config['left_extension'] = config['right_extension'] = np.floor(config['event_separation'] / 2) - 1
        if config['event_separation'] <= config['left_extension'] + config['right_extension']:
            raise ValueError("event_separation must be larger than left + right extension "
                             "to avoid data duplication.")
        config['left_extension'] = abs(config['left_extension'])            # Deal with Chris' shenanigans

        self.config = config

        log.info("Starting trigger")
        log.info("\tEvent separation threshold: %0.2f ms", config['event_separation'] / units.ms)
        log.info("\tSignal separation threshold: %0.2f us", config['signal_separation'] / units.us)
        log.info("\tLeft extension: %0.2f us", config['left_extension'] / units.us)
        log.info("\tRight extension: %0.2f us", config['right_extension'] / units.us)

        # Initialize buffer for numba signal finding routine: we must initialize a large buffer here
        # since we can't create / extend arrays from numba. I don't want to implement some code which lets
        # the numba routine send some message code for buffer_full, in which case it is called again, etc.
        # TriggerSignal array is about 100 bytes per signal, so 5M signals is 500 MB RAM.
        # Could double it, not increase by factor ten.
        self.numba_signals_buffer = np.zeros(config.get('numba_signal_buffer_size', int(5e6)),
                                             dtype=TriggerSignal.get_dtype())

        # Convert trigger_probability dictionary of dictionaries to 2d array, with intermediate values filled in
        self.log.info("\tTrigger probabilities: %s" % str(config['trigger_probability']))
        p_length = max([max(ps.keys()) for ps in config['trigger_probability'].values()]) + 1
        p_matrix = np.zeros((3, p_length), dtype=np.float)
        for sig_type, ps in config['trigger_probability'].items():
            for i, n in enumerate(sorted(ps.keys())):
                p_matrix[sig_type][n:] = ps[n]
        self.p_matrix = p_matrix

    def add_new_data(self, times, last_time_searched):
        """Adds more data to the trigger's buffer"""
        self.log.debug("Received %d more times" % len(times))
        self.times = np.concatenate((self.times, times))
        self.last_time_searched = last_time_searched
        self.pulses_read += len(times)

    def get_trigger_ranges(self):
        """Yield successive trigger ranges from the trigger's buffer.
        raises StopIteration if insufficient data to continue.
        """
        times = self.times
        config = self.config

        # Find signals. This happens in numba, so it should be super-fast.d
        n_signals_found = find_signals(times=times,
                                       signal_separation=self.config['signal_separation'],
                                       signal_buffer=self.numba_signals_buffer)
        signals = self.numba_signals_buffer[:n_signals_found]
        self.log.debug("Trigger found %d signals in data" % n_signals_found)

        # Classify the signals; modifies signals in-place.
        classify_signals(signals, s1_max_rms=self.config['s1_max_rms'], s2_min_pulses=self.config['s2_min_pulses'])

        # Determine which signals trigger an event; modifies signals in-place
        decide_triggers(signals, p_matrix=self.p_matrix)

        # Group the triggers into event ranges
        trigger_times = signals[signals['trigger']]['left_time']
        event_ranges = self.find_event_ranges(trigger_times)
        self.log.debug("Found %d event ranges" % len(event_ranges))

        # What data can we clear from the times buffer?
        clear_until = self.last_time_searched - config['event_separation']

        if len(event_ranges) and self.more_data_is_coming and event_ranges[-1][-1] > clear_until:
            # The last event range can still be extended by future data, so we should not send it yet.
            # We should also not clear data in what will become part of that event.
            self.log.debug("Not sending last event range, might be extended by future data")
            clear_until = min(clear_until, event_ranges[-1][0] - config['left_extension'])
            event_ranges = event_ranges[:-1]

        # Group the signals with the events,
        # It's ok to do a for loop in python over the events, that happens anyway for sending events out
        # signal_indices_buffer will hold, for each event, the start and stop (inclusive) index of signals in the event
        signal_indices_buffer = np.zeros((len(event_ranges), 2), dtype=np.int)
        signal_is_in_event = np.zeros(len(signals), dtype=np.bool)

        if len(event_ranges):
            group_signals(signals, event_ranges, signal_indices_buffer)     # Modifies signal_indices_buffer in-place
            for event_i in range(len(event_ranges)):
                signal_start_i, signal_end_i = signal_indices_buffer[event_i]
                # Notice the + 1 for python's exclusive indexing below...
                signal_is_in_event[signal_start_i:signal_end_i + 1] = True
                yield event_ranges[event_i], signals[signal_start_i:signal_end_i + 1]
                self.events_built += 1
                self.total_event_length += event_ranges[event_i][1] - event_ranges[event_i][0]

        # Store the signals that were not in any events in an attribute, so people can get them out
        self.signals_beyond_events = signals[True ^ (signal_is_in_event)]

        # Clear times (safe range was determined above)
        # if needed, this can be sped up a lot by a numba search routine
        self.log.debug("Clearing times after %d" % clear_until)
        self.times = times[times >= clear_until]

    def find_event_ranges(self, trigger_times):
        """Return event ranges corresponding to trigger_times"""
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

    def get_end_of_run_info(self):
        """Return a dictionary with end-of-run information, for printing or for the runs database"""
        events_built = self.events_built
        mean_event_length = self.total_event_length / events_built if events_built else 0
        return {'events_built': events_built,
                'mean_event_length': mean_event_length,
                'last_time_searched': self.last_time_searched,
                'timestamp': time.time(),
                'pulses_read': self.pulses_read,
                'pax_version': pax.__version__,
                'config': self.config}


@numba.jit(nopython=True)
def find_signals(times, signal_separation, signal_buffer):
    """Fill signal_buffer with signals in times. Returns number of signals found so you can slice the buffer.
    signal_separation: group pulses into signals separated by signal_separation.
    Online RMS algorithm is Knuth/Welford: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    # does_channel_contribute = np.zeros(1000, dtype=np.bool)   # TODO: get n_channels from config
    in_signal = False
    passes_test = False
    current_signal = 0
    M2 = 0.0      # Temporary variable for online RMS computation

    for time_index, t in enumerate(times):

        last_time = time_index == len(times) - 1
        if not last_time:
            passes_test = times[time_index+1] - t < signal_separation

        if not in_signal and passes_test:
            # Start a signal. Note we must set ALL attributes to clear potential mess from the buffer.
            # I wish numpy arrays could grow automatically... but then they would probably not be fast...
            in_signal = True
            signal_buffer[current_signal].left_time = t
            signal_buffer[current_signal].right_time = 0
            signal_buffer[current_signal].time_mean = 0
            signal_buffer[current_signal].time_rms = 0
            signal_buffer[current_signal].n_pulses = 0
            signal_buffer[current_signal].n_contributing_channels = 0
            signal_buffer[current_signal].area = 0
            signal_buffer[current_signal].x = 0
            signal_buffer[current_signal].y = 0

        if in_signal:                           # Notice if, not elif. Work on first time in signal too.
            # Update signal quantities
            signal_buffer[current_signal].n_pulses += 1
            s = signal_buffer[current_signal]
            delta = t - s.time_mean
            signal_buffer[current_signal].time_mean += delta / s.n_pulses
            M2 += delta * (t - s.time_mean)     # Notice NOT **2, s.time_mean has changed, this is important.

            if not passes_test or last_time:
                # Store current signal, move on to next
                signal_buffer[current_signal].right_time = t
                signal_buffer[current_signal].time_rms = (M2 / signal_buffer[current_signal].n_pulses)**0.5
                current_signal += 1
                M2 = 0
                in_signal = False

    return current_signal


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


@numba.jit(nopython=True)
def group_signals(signals, event_ranges, signal_indices_buffer):
    """Fill signal_indices_buffer with array of (left, right) indices
    indicating which signals belong with event_ranges.
    """
    current_event = 0
    in_event = False
    signals_start = 0

    for signal_i, signal in enumerate(signals):
        if in_event:
            if signal['left_time'] > event_ranges[current_event, 1] or signal_i == len(signals) - 1:
                # Signal is the last in the current event, yield and move to new event
                signal_indices_buffer[current_event][0] = signals_start
                signal_indices_buffer[current_event][1] = signal_i
                in_event = False
                current_event += 1
                if current_event > len(event_ranges) - 1:
                    # Done with all events, rest of signals is outside events
                    break
        else:
            if signal['left_time'] >= event_ranges[current_event, 0]:
                # Signal is the first in the current event
                in_event = True
                signals_start = signal_i
