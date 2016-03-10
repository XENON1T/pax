"""XENON1T trigger controller and low-level trigger

The low-level trigger turns the raw data (or rather, the pulse start times) into TriggerSignals.
A TriggerSignal is any range of pulse start times no further than signal_window apart from each other.

The TriggerSignals are passed to the high-level trigger modules, most importantly the main trigger.
This main trigger classifies the TriggerSignals into S1s and S2s and takes care of proper range extension of events.

The low-level trigger also does concurrent monitoring: some highly reduced information about the data is continuously
saved to a separate hdf5 data file. This contains:
 - The pmt dark rate, or rather, the number of pulses in each PMT beyond TriggerSignals (sampled at regular intervals)
 - The number of TriggerSignals with just two contributing PMTs (sampled at regular, but less frequent intervals),
   for each possible pair of PMTs.
 - (if desired) the TriggerSignals outside the events, or a subset of them. This is currently an experimental feature,
   use with caution as it can slow the trigger down a lot. Note the full raw data is not saved for these signals,
   just the TriggerSignal information (see pax.datastructure).

"""
import time
import logging
import os

import numba
import numpy as np
import h5py

import pax          # For version number
from pax.datastructure import TriggerSignal


# Interrupts thrown by the signal finding code
# Negative, since positive numbers indicate number of signals found during normal operation
SIGNAL_BUFFER_FULL = -1
SAVE_DARK_MONITOR_DATA = -2


class Trigger(object):
    """The XENON1T trigger controller"""
    more_data_is_coming = True
    dark_monitor_data_saves = 0     # How often did we save dark monitor data since the last full save?

    # Buffers for data taken so far. Will be extended as needed.
    # TODO: annoying to have separate buffers, better to change to new record array dtype
    times = np.zeros(0, dtype=np.int64)
    channels = np.zeros(0, dtype=np.int64)
    modules = np.zeros(0, dtype=np.int64)
    last_time_searched = 0

    # Statistics for dumping at end of run
    events_built = 0
    total_event_length = 0
    pulses_read = 0
    signals_found = 0
    triggers_found = 0

    def __init__(self, config, pmt_data):
        self.log = logging.getLogger('Trigger')
        self.config = config

        # Initialize buffer for numba signal finding routine: we must initialize a large buffer here
        # since we can't create / extend arrays from numba. I don't want to implement some code which lets
        # the numba routine send some message code for buffer_full, in which case it is called again, etc.
        self.numba_signals_buffer = np.zeros(config['numba_signal_buffer_size'],
                                             dtype=TriggerSignal.get_dtype())

        # Build a (module, channel) ->  lookup matrix
        # I whish numba had some kind of dictionary / hashtable support...
        # but this will work as long as the module serial numbers are small :-)
        # I will asssume always and everywhere the pmt position numbers start at 0 and increase by 1 continuously!
        # Initialize the matrix to n_channels, which is one above the last PMT
        # This will ensure we do not crash on data in 'ghost' channels (not plugged in,
        # do report data in self-triggered mode)
        n_channels = len(pmt_data)
        max_module = max([q['digitizer']['module'] for q in pmt_data])
        max_channel = max([q['digitizer']['channel'] for q in pmt_data])
        self.pmt_lookup = n_channels * np.ones((max_module + 1, max_channel + 1), dtype=np.int)
        for q in pmt_data:
            module = q['digitizer']['module']
            channel = q['digitizer']['channel']
            self.pmt_lookup[module][channel] = q['pmt_position']

        # Build the coincidence tally matrix
        # Reason for +1 is again 'ghost' channels, see comment above
        self.coincidence_tally = np.zeros((len(pmt_data) + 1, len(pmt_data) + 1), dtype=np.int)

        # Create file & datasets for extra trigger data ('dark monitor')
        dir = os.path.dirname(self.config['trigger_data_filename'])
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        self.f = h5py.File(self.config['trigger_data_filename'], mode='w')
        self.dark_rate_dataset = self.f.create_dataset('dark_pulses',
                                                       shape=(0, n_channels + 1),
                                                       maxshape=(None, n_channels + 1),
                                                       dtype=np.int,
                                                       compression="gzip")
        self.dark_rate_dataset.attrs['save_interval'] = self.config['dark_rate_save_interval']
        self.coincidence_rate_dataset = self.f.create_dataset('coincidence_tally',
                                                              shape=(0, n_channels + 1, n_channels + 1),
                                                              maxshape=(None, n_channels + 1, n_channels + 1),
                                                              dtype=np.int,
                                                              compression="gzip")
        self.dark_rate_dataset.attrs['save_interval'] = self.config['dark_rate_save_interval'] * \
                                                        self.config['dark_monitor_full_save_every']
        if self.config['save_signals_outside_events']:
            self.outside_signals_dataset = self.f.create_dataset('signals_outside_events',
                                                                 shape=(0,),
                                                                 maxshape=(None,),
                                                                 dtype=self.numba_signals_buffer.dtype,
                                                                 compression="gzip")

    def add_new_data(self, times, last_time_searched, channels=None, modules=None):
        """Adds more data to the trigger's buffer"""
        # Support for adding data without channel & module info
        if channels is None:
            channels = np.zeros(len(times), dtype=np.int32)
        if modules is None:
            modules = np.zeros(len(times), dtype=np.int32)
        self.log.debug("Received %d more times, %d times already in buffer" % (len(times), len(self.times)))
        self.times = np.concatenate((self.times, times))
        self.channels = np.concatenate((self.channels, channels))
        self.modules = np.concatenate((self.modules, modules))
        self.last_time_searched = last_time_searched
        self.pulses_read += len(times)

    def save_dark_monitor_data(self, last_time=False):
        self.log.debug("Saving PMT dark date, total %d pulses" % np.trace(self.coincidence_tally))
        # Save the PMT dark rate
        h5py_append(self.dark_rate_dataset, np.diagonal(self.coincidence_tally))
        self.dark_monitor_data_saves += 1
        np.fill_diagonal(self.coincidence_tally, 0)         # Clear the dark rate counters
        if last_time or self.dark_monitor_data_saves == self.config['dark_monitor_full_save_every']:
            # Save the full coincidence rate
            self.log.debug("Saving coincidence tally matrix, total %d" % self.coincidence_tally.sum())
            h5py_append(self.coincidence_rate_dataset, self.coincidence_tally)
            self.dark_monitor_data_saves = 0
        self.coincidence_tally *= 0

    def run(self):
        """Yield successive trigger ranges from the trigger's buffer.
        raises StopIteration if insufficient data to continue.
        """
        times = self.times
        config = self.config

        ##
        # Low-level trigger: find TriggerSignals.
        ##
        # Get the signal finder generator
        does_channel_contribute = np.zeros(len(self.coincidence_tally), dtype=np.int)
        sigf = signal_finder(times=self.times,
                             channels=self.channels,
                             modules=self.modules,
                             signal_separation=self.config['signal_separation'],
                             signal_buffer=self.numba_signals_buffer,
                             coincidence_tally=self.coincidence_tally,
                             pmt_lookup=self.pmt_lookup,
                             does_channel_contribute=does_channel_contribute,
                             dark_rate_save_interval=self.config['dark_rate_save_interval'])
        saved_buffers = []
        while True:
            result = next(sigf)
            if result >= 0:
                # All times processed
                n_signals_found = result
                if len(saved_buffers):
                    self.log.debug("Signal finder done on this range. Some previous signal buffers were saved, "
                                   "concatenating and returning them.")
                    saved_buffers.append(self.numba_signals_buffer[:n_signals_found])
                    signals = np.concatenate(saved_buffers)
                else:
                    signals = self.numba_signals_buffer[:n_signals_found]
                break
            elif result == SIGNAL_BUFFER_FULL:
                self.log.debug("Signal buffer is full, copying it out.")
                # We can't do this in numba because it involves growing a list.
                # (maybe we could use np.resize though... Never mind)
                saved_buffers.append(self.numba_signals_buffer.copy())
            elif result == SAVE_DARK_MONITOR_DATA:
                # Save and clear the coincididence & dark rate tally
                self.save_dark_monitor_data()
            else:
                raise ValueError("Unknown signal finder interrupt %d!" % result)

        # Could the last_search_time be in the middle of a signal?
        # If so, and more data is coming, we have to retain some times for the next pass.
        # TODO





        self.signals_found += len(signals)
        self.log.debug("Low-level trigger found %d signals in data." % len(signals))

        ##
        # High-level trigger: run signals by each high-level trigger module to get final event ranges.
        # The main HLT module classifies the signals, which other triggers can make use of or overwrite.
        ##

        # First run the HLTs. This sets the lookback times if needed
        ranges_per_module = {}
        for hlt in self.hlt_modules:
            ranges_per_module[hlt.name] = hlt.get_event_ranges(signals)

        if self.more_data_is_coming:
            # Some HLTs may want to revisit the data when more comes in, so we have to keep some back.
            # Triggers already found in this lookback time (i.e. that end within it) cannot be sent.
            # They will be recomputed from the data next time.
            clear_until = min([hlt.lookback_time for hlt in self.hlt_modules])
            for k, v in ranges_per_module:
                ranges_per_module[k] = v[v[:, 1] < clear_until]

        # TODO: Concatenate and sort the event ranges by start time
        n_events = sum([len(v) for v in ranges_per_module.values()])

        self.triggers_found += len(trigger_times)
        self.log.debug("Found %d event ranges" % len(event_ranges))

        # What data can we clear from the times buffer?
        clear_until = self.last_time_searched - config['event_separation']


        # Group the signals with the events,
        # It's ok to do a for loop in python over the events, that happens anyway for sending events out
        # signal_indices_buffer will hold, for each event, the start and stop (inclusive) index of signals in the event
        signal_indices_buffer = np.zeros((len(event_ranges), 2), dtype=np.int)

        if len(event_ranges):
            group_signals(signals, event_ranges, signal_indices_buffer)     # Modifies signal_indices_buffer in-place
            for event_i in range(len(event_ranges)):
                signal_start_i, signal_end_i = signal_indices_buffer[event_i]
                # Notice the + 1 for python's exclusive indexing below...
                yield event_ranges[event_i], signals[signal_start_i:signal_end_i + 1]
                self.events_built += 1
                self.total_event_length += event_ranges[event_i][1] - event_ranges[event_i][0]

        # Save signals outside event, if desired
        # This can slow the trigger down a bit, so it is in a second loop
        # TODO: if needed, this can probably be sped up a lot by using a numba routine
        if self.config['save_signals_outside_events']:
            signal_is_in_event = np.zeros(len(signals), dtype=np.bool)
            for event_i in range(len(event_ranges)):
                signal_start_i, signal_end_i = signal_indices_buffer[event_i]
                signal_is_in_event[signal_start_i:signal_end_i + 1] = True

            outsigs = signals[(True ^ signal_is_in_event)]
            outsigs = np.concatenate([outsigs[(outsigs['type'] == sigtype) &
                                              (outsigs['n_pulses'] >=
                                               self.config['outside_signals_save_thresholds'][sigtype])]
                                      for sigtype in range(2 + 1)])
            # Don't record signals in overlap bit as outside
            # TODO: Fix more nicely!
            outsigs = outsigs[outsigs['left_time'] < clear_until]

            # Store basic info about these signals in an hdf5.
            # For now, make them available in attribute as well, so other code can grab it.
            self.signals_outside_events = outsigs
            if len(outsigs):
                self.log.debug("Storing %d signals outside events" % len(outsigs))
                h5py_append(self.outside_signals_dataset, outsigs)

        # Clear times (safe range was determined above)
        # TODO: if needed, this can probably be sped up a lot by a numba search routine
        self.log.debug("Clearing times after %d" % clear_until)
        self.times = times[times >= clear_until]


    def shutdown(self):
        """Shuts down trigger, return a dictionary with end-of-run information, for printing or for the runs database
        Will close outside_signals_file
        """
        self.save_dark_monitor_data(last_time=True)
        events_built = self.events_built
        mean_event_length = self.total_event_length / events_built if events_built else 0
        if hasattr(self, 'f'):
            self.f.close()

        end_info = {'events_built': events_built,
                    'mean_event_length': mean_event_length,
                    'last_time_searched': self.last_time_searched,
                    'timestamp': time.time(),
                    'pax_version': pax.__version__}
        for attrname in ('config', 'pulses_read', 'signals_found', 'triggers_found'):
            end_info[attrname] = getattr(self, attrname)

        return end_info


@numba.jit(nopython=True)
def signal_finder(times, channels, modules, signal_separation, signal_buffer, coincidence_tally,
                  pmt_lookup,
                  does_channel_contribute,
                  dark_rate_save_interval):
    """Fill signal_buffer with signals in times. Arguments:
     - channels, modules: same length as times, indices in pmt_lookup which
     - signal_separation: group pulses into signals separated by signal_separation.
     - coincidence_tally: nxn matrix of zero where n is number of channels,used to store 2-pmt coincidences
       (with 1-pmt, i.e. dark rate, on diagonal)
     - pmt_lookup: lookup matrix for pmt numbers. First index is digitizer module, second is digitizer channel
     - dark_rate_save_interval: yield SAVE_DARK_MONITOR every dark_rate_save_interval
     - does channel contibute: zero array of len n_channels. Very annoying we can't allocate this inside!
    Online RMS algorithm is Knuth/Welford: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    in_signal = False
    passes_test = False     # Does the current time pass the signal inclusion test?
    current_signal = 0      # Index of the current signal in the signal buffer
    M2 = 0.0                # Temporary variable for online RMS computation
    if len(times):
        next_save_time = times[0] + dark_rate_save_interval

    for time_index, t in enumerate(times):

        if t > next_save_time:
            yield SAVE_DARK_MONITOR_DATA
            next_save_time += dark_rate_save_interval

        pmt = pmt_lookup[modules[time_index], channels[time_index]]

        # Should this time be in a signal -- is the next time close enough?
        is_last_time = time_index == len(times) - 1
        if not is_last_time:
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
            does_channel_contribute *= 0

        if in_signal:                           # Notice if, not elif. Work on first time in signal too.
            # Update signal quantities
            does_channel_contribute[pmt] = 1
            signal_buffer[current_signal].n_pulses += 1
            delta = t - signal_buffer[current_signal].time_mean
            signal_buffer[current_signal].time_mean += delta / signal_buffer[current_signal].n_pulses
            # Notice the below line does not have delta **2. The time_mean changed in the previous line!
            M2 += delta * (t - signal_buffer[current_signal].time_mean)

            if not passes_test or is_last_time:
                # Signal has ended: store its quantities
                signal_buffer[current_signal].right_time = t
                signal_buffer[current_signal].time_rms = (M2 / signal_buffer[current_signal].n_pulses)**0.5
                signal_buffer[current_signal].n_contributing_channels = does_channel_contribute.sum()
                if signal_buffer[current_signal].n_contributing_channels == 2:
                    # Update coincidence tally
                    # Life is hard inside nopython=True...
                    first_channel = -999
                    second_channel = -999
                    for i in range(len(does_channel_contribute)):
                        if does_channel_contribute[i]:
                            first_channel = i
                            break
                    for i in range(first_channel + 1, len(does_channel_contribute)):
                        if does_channel_contribute[i]:
                            second_channel = i
                            break
                    coincidence_tally[first_channel, second_channel] += 1

                current_signal += 1
                M2 = 0
                in_signal = False

                if current_signal == len(signal_buffer):
                    yield SIGNAL_BUFFER_FULL
                    # Caller will now have copied & cleared signal buffer, can start from 0
                    current_signal = 0

        else:
            # Update dark rate tally
            coincidence_tally[pmt][pmt] += 1

    # Let caller know number of signals found, then raise StopIteration
    yield current_signal


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


def h5py_append(dataset, records):
    """Append records to h5py dataset, resizing axis=0 as needed
    This probably goes completely against the philosophy of hdf5 and is super convenient
    """
    if len(dataset.shape) != len(records.shape):
        # Hack for appending a single record
        want_shape = list(dataset.shape)
        want_shape[0] = -1
        records = records.reshape(tuple(want_shape))
    orig_size = dataset.shape[0]
    dataset.resize(orig_size + records.shape[0], axis=0)
    dataset[orig_size:] = records


class TriggerModule(object):
    """Base class for high-level trigger modules"""

    # Earliest time the trigger wants to revisit later when more data comes in
    # Not all triggers need this; the main trigger does.
    lookback_time = float('inf')

    # Unique integer ID for each trigger. This must never change!
    numeric_id = -1

    def __init__(self, trigger, config):
        self.trigger = trigger
        self.config = config
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.__class__.__name__)
        self.startup()

    def _get_event_ranges(self, signals):

        # TODO: common handling code for lookback & signal retention

        event_ranges = self._get_event_ranges(signals)
        # Add the HLT number to the event ranges, then return them
        return np.concatenate((event_ranges,
                               np.ones(len(event_ranges)) * self.numeric_id),
                              axis=1)

    def get_event_ranges(self, signals):
        raise NotImplementedError

    def startup(self):
        self.log.debug("%s did not define a startup." % self.name)

    def shutdown(self):
        self.log.debug("%s did not define a shutdown." % self.name)