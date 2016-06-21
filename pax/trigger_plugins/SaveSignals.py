import numba
import numpy as np
from pax.trigger import TriggerPlugin
from pax.exceptions import TriggerGroupSignals


class SaveSignals(TriggerPlugin):

    def process(self, data):
        is_in_event = np.zeros(len(data.signals), dtype=np.bool)

        if len(data.signals) and len(data.event_ranges):
            # sig_group_ind will hold, for each event,the start and stop (inclusive) index of signals
            sig_idx = np.zeros((len(data.event_ranges), 2), dtype=np.int)
            group_signals(data.signals, data.event_ranges, sig_idx, is_in_event)

            # It's ok to do a for loop in python over the events here,
            # there's a python loop anyway for sending events out
            signals_by_event = []
            for event_i in range(len(data.event_ranges)):
                signals_by_event.append(data.signals[sig_idx[event_i][0]:sig_idx[event_i][1] + 1])
            data.signals_by_event = signals_by_event

        save_mode = self.config.get('save_signals')
        if save_mode:
            sigs = data.signals
            if self.config.get('only_save_signals_outside_events'):
                sigs = sigs[True ^ is_in_event]
            if save_mode == 'full':
                sigs = sigs[sigs['n_pulses'] >= self.config['signals_save_threshold']]
                if len(sigs):
                    self.log.debug("Storing %d signals in trigger data" % len(sigs))
                    self.trigger.save_monitor_data('trigger_signals', sigs)
            elif save_mode == '2d_histogram':
                hist, _, _ = np.histogram2d(np.log10(sigs['n_pulses']),
                                            np.log10(sigs['time_rms'] + 1),
                                            bins=(np.linspace(0, 10, 100),
                                                  np.linspace(0, 10, 100)))
                self.trigger.save_monitor_data('trigger_signals_histogram', hist)


@numba.jit(nopython=True)
def group_signals(signals, event_ranges, signal_indices_buffer, is_in_event):
    """Fill signal_indices_buffer with array of (left, right) indices
    indicating which signals belong in which event range.
    is_in_event will be set to True is a signal is in an event.
    """
    current_event = 0
    in_event = False
    signals_start = 0

    for signal_i, signal in enumerate(signals):
        if not in_event:
            if signal['left_time'] >= event_ranges[current_event, 0]:
                if signal['left_time'] > event_ranges[current_event, 1]:
                    raise TriggerGroupSignals("Error during signal grouping: event without signals??")
                # Signal is the first in the current event
                in_event = True
                signals_start = signal_i

        if in_event:            # Notice no elif, in_event may just have been set
            is_in_event[signal_i] = True
            if signal_i == len(signals) - 1 or signals[signal_i + 1]['left_time'] > event_ranges[current_event, 1]:
                # This signal is the last in the current event, yield and move to new event
                signal_indices_buffer[current_event][0] = signals_start
                signal_indices_buffer[current_event][1] = signal_i
                in_event = False
                current_event += 1
                if current_event > len(event_ranges) - 1:
                    # Done with all events, rest of signals are outside events
                    return
