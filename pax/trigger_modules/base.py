import logging

import numba
import numpy as np


class TriggerModule(object):
    """Base class for high-level trigger modules"""

    # Unique integer ID for each trigger. If your trigger module is committed to pax, this can never change again!
    numeric_id = -1

    def __init__(self, trigger, config):
        if self.numeric_id == -1:
            raise RuntimeError("Attempt to run the TriggerModule base class directly, or numeric trigger id not set")
        self.trigger = trigger
        self.config = config
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.__class__.__name__)
        self.end_of_run_info = dict(events_built=0,
                                    total_event_length=0)
        self.startup()

    def get_event_ranges(self, signals):
        raise NotImplementedError

    def startup(self):
        self.log.debug("%s did not define a startup." % self.name)

    def shutdown(self):
        self.log.debug("%s did not define a shutdown." % self.name)

    def make_events(self, event_ranges, signals):
        """Yields events from event_ranges, grabbing the right signals which are in each event range along with it
        and updating the end of run statistics.
        """
        if not len(event_ranges):
            # Don't send out any events
            # Code below won't work if events = [], since resulting numpy array is not yet two dimensional
            raise StopIteration
        self.log.debug("Sending %d events" % len(event_ranges))

        # Convert from list to numpy array if needed
        if isinstance(event_ranges, list):
            event_ranges = np.array(event_ranges, dtype=np.int64)

        # Group the signals with the event ranges
        # self.signal_indices_buffer will hold, for each event,the start and stop (inclusive) index of signals
        # It's stored in an attribute since it may be of interest to code that runs after it
        # (for example, the main trigger uses it to know which signals are outside any events)
        self.signal_indices_buffer = np.zeros((len(event_ranges), 2), dtype=np.int)
        group_signals(signals, event_ranges, self.signal_indices_buffer)

        # Group the signals with the events, then send out event range, signal info, trigger id foreach event
        # It's ok to do a for loop in python over the events, we're in a python loop anway for sending events out
        for event_i, (start, stop) in enumerate(event_ranges):
            signal_start_i, signal_end_i = self.signal_indices_buffer[event_i]
            yield event_ranges[event_i], signals[signal_start_i:signal_end_i + 1], self.numeric_id
            self.end_of_run_info['events_built'] += 1
            self.end_of_run_info['total_event_length'] += stop - start      # No +1 here, last sample is inclusive.


@numba.jit(nopython=True)
def group_signals(signals, event_ranges, signal_indices_buffer):
    """Fill signal_indices_buffer with array of (left, right) indices
    indicating which signals belong in which event range.
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
