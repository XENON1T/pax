import numba
import numpy as np
from pax.trigger import TriggerPlugin, h5py_append
from pax.datastructure import TriggerSignal


class SaveSignals(TriggerPlugin):

    def startup(self):
        if self.config['save_signals_outside_events']:
            f = self.trigger.dark_monitor_data_file
            self.outside_signals_dataset = f.create_dataset('signals_outside_events',
                                                            shape=(0,),
                                                            maxshape=(None,),
                                                            dtype=TriggerSignal.get_dtype(),
                                                            compression="gzip")

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

        if self.config['save_signals_outside_events']:
            outsigs = data.signals[True ^ is_in_event]
            if len(outsigs):
                self.log.debug("Storing %d signals outside events" % len(outsigs))
                h5py_append(self.outside_signals_dataset, outsigs)


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
