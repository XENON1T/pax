import numpy as np
import numba

from pax.trigger import TriggerPlugin, pulse_dtype
from pax.datastructure import TriggerSignal


class HandleEdgePulses(TriggerPlugin):

    saved_pulses = np.array([], dtype=pulse_dtype)

    def process(self, data):
        if len(self.saved_pulses):
            self.log.debug("Injecting %d saved pulses" % len(self.saved_pulses))
            data.pulses = np.concatenate((self.saved_pulses, data.pulses))
            self.saved_pulses = self.saved_pulses[:0]

        if not data.last_data:
            # We may not be able to look at all the added pulses yet, since the next data batch of data
            # could change their interpretation. Find the last index that is safe too look at.
            last_i = find_last_break(times=data.pulses['time'],
                                     last_time=data.last_time_searched,
                                     break_time=self.config['signal_separation'])

            # Keep the pulses we can work with, save the rest in for the next time we are called.
            self.saved_pulses = data.pulses[last_i + 1:]
            data.pulses = data.pulses[:last_i + 1]

        self.log.debug("Saved %d pulses for later" % len(self.saved_pulses))
        data.batch_info_doc['pulses_saved_for_next_batch'] = len(self.saved_pulses)


class HandleEdgeSignals(TriggerPlugin):

    saved_signals = np.array([], dtype=TriggerSignal.get_dtype())

    def process(self, data):
        if len(self.saved_signals):
            self.log.debug("Injecting %d saved signals" % len(self.saved_signals))
            data.signals = np.concatenate((self.saved_signals, data.signals))
            # Somehow we really need to re-sort here, or we get in big trouble (TriggerGroupSignals exceptions)
            # TODO: Why?? This probably indicates a problem somewhere else...
            data.signals.sort(order='lef_time')
            self.saved_signals = self.saved_signals[:0]

        if not data.last_data:
            # We may not be able to look at all the added signals yet, since the next data batch of data
            # could change their interpretation.
            lookback_time = data.last_time_searched - self.config['event_separation']

            # Can some triggers be grouped with potential triggers in the next batch of data?
            trigger_times = data.signals[data.signals['trigger']]['left_time']
            last_trigger_i = find_last_break(times=trigger_times,
                                             last_time=data.last_time_searched,
                                             break_time=self.config['event_separation'])
            if last_trigger_i != len(trigger_times) - 1:
                lookback_time = trigger_times[last_trigger_i + 1] - self.config['left_extension']

            # What is the last signal we can work with?
            # TODO: Can be sped up by loop in numba. But probably doesn't matter much, should only run a few iterations.
            i = 0
            for i, t in enumerate(data.signals['left_time'][::-1]):
                if t < lookback_time:
                    break
            last_ok_index = len(data.signals) - 1 - i
            self.saved_signals = data.signals[last_ok_index + 1:]
            data.signals = data.signals[:last_ok_index + 1]

        self.log.debug("Saved %d signals for next batch" % len(self.saved_signals))
        data.batch_info_doc['signals_saved_for_next_batch'] = len(self.saved_signals)


@numba.jit(nopython=True)
def find_last_break(times, last_time, break_time):
    """Return the last index in times after which there is a gap >= break_time.
    If the last entry in times is further than signal_separation from last_time,
    that last index in times is returned.
    Returns -1 if no break exists anywhere in times.
    """
    imax = len(times) - 1
    # Start from the end of the list, iterate backwards
    for _i in range(len(times)):
        i = imax - _i
        t = times[i]
        if t < last_time - break_time:
            return i
        else:
            last_time = t
    return -1
