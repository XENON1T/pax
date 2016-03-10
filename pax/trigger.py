import inspect
import time
import logging
import os
from glob import glob

import numba
import numpy as np
import h5py

import pax          # For version number
from pax.datastructure import TriggerSignal
from pax.utils import PAX_DIR
from pax.trigger_modules.base import TriggerModule

times_dtype = np.dtype([('time', np.int64),
                        ('pmt', np.int32),
                        # ('area', np.float64)
                        ])

# Interrupts thrown by the signal finding code
# Negative, since positive numbers indicate number of signals found during normal operation
SIGNAL_BUFFER_FULL = -1
SAVE_DARK_MONITOR_DATA = -2


class Trigger(object):
    """XENON1T trigger controller and low-level trigger

    The low-level trigger turns the raw data (or rather, the pulse start times) into TriggerSignals.
    A TriggerSignal is any range of pulse start times no further than signal_window apart from each other.

    The TriggerSignals are passed to the high-level trigger modules, most importantly the main trigger.
    This main trigger classifies the TriggerSignals into S1s and S2s, then triggers on them as required.
    Other trigger modules can e.g. trigger at specific times, or if some completely different conditions are met.

    Some highly reduced information about the data is continuously saved to a separate hdf5 data file:
     - The pmt dark rate, or rather, the number of pulses in each PMT beyond TriggerSignals (sampled regularly)
     - The number of TriggerSignals with just two contributing PMTs (sampled at regular, but less frequent intervals),
       for each possible pair of PMTs.
    The high-level triggers can add more information to this, see e.g. the main trigger docs.

    Additionally, some information is written to the runs database at the end of processing a run, e.g. the number of
    signals found. High-level triggers can also add additional info to this, e.g. the number of events found.
    """
    more_data_is_coming = True
    dark_monitor_data_saves = 0     # How often did we save dark monitor data since the last full save?

    # Buffers for data taken so far. Will be extended as needed.
    # TODO: annoying to have separate buffers, better to change to new record array dtype
    times = np.zeros(0, dtype=times_dtype)
    last_time_searched = 0

    def __init__(self, pax_config):
        self.log = logging.getLogger('Trigger')
        self.config = pax_config['Trigger']
        pmt_data = pax_config['DEFAULT']['pmts']
        self.end_of_run_info = dict(times_read=0,
                                    signals_found=0,
                                    start_timestamp=time.time(),
                                    pax_version=pax.__version__,
                                    config=self.config)

        # Initiaize the high-level triggers we should run
        # First build a dictionary mapping hlt names to hlt classes, then initialize the required classes
        hlt_classes = {}
        for module_filename in glob(os.path.join(PAX_DIR + '/trigger_modules/*.py')):
            module_name = os.path.splitext(os.path.basename(module_filename))[0]
            if module_name.startswith('_'):
                continue

            # Import the module, after which we can do pax.trigger_modules.module_name
            __import__('pax.trigger_modules.%s' % module_name, globals=globals())

            # Now get all the high-level triggers defined in each module
            for hlt_name, hlt_class in inspect.getmembers(getattr(pax.trigger_modules, module_name),
                                                          lambda x: type(x) == type and issubclass(x, TriggerModule)):
                if hlt_name == 'TriggerModule':
                    continue
                if hlt_name in hlt_classes:
                    raise ValueError("Two TriggerModule's named %s!" % hlt_name)
                hlt_classes[hlt_name] = hlt_class

        # Finally, we can initialize the high-level trigger modules specified in the configuration
        self.hlts = []
        for _, hlt_name in self.config['high_level_trigger_modules']:
            if hlt_name not in hlt_classes:
                raise ValueError("Don't know a high-level trigger called %s!" % hlt_name)
            self.hlts.append(hlt_classes[hlt_name](config=pax_config.get('Trigger.%s' % hlt_name, {})))

        # Initialize buffer for numba signal finding routine.
        # Although we're able to extend this buffer as needed, we must do so outside numba
        # and it involves copyping data, so if you pick too small a buffer size you will hurt performance.
        self.numba_signals_buffer = np.zeros(self.config['numba_signal_buffer_size'],
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

    def add_new_data(self, start_times, last_time_searched, channels=None, modules=None):
        """Adds more data to the trigger's buffer"""
        times = np.zeros(len(start_times), dtype=times_dtype)
        times['time'] = start_times
        # Convert the channel/module specs into pmt numbers.
        # If not specified, pretend everything is from the 'ghost' pmt.
        if channels is not None and modules is not None:
            get_pmt_numbers(channels, modules, pmts_buffer=times['pmt'], pmt_lookup=self.pmt_lookup)
        else:
            times['pmt'] = len(self.coincidence_tally) - 1
        self.log.debug("Received %d more times, %d times already in buffer" % (len(times), len(self.times)))
        self.times = np.concatenate((self.times, times))
        self.last_time_searched = last_time_searched
        self.end_of_run_info['times_read'] += len(times)

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
        """Run the XENON1T trigger on the previously added data.
        Yields successively: (start, stop), signals_in_event
        raises StopIteration if insufficient data to continue.
        """
        if self.more_data_is_coming:
            # We may not be able to look at all the added times yet, since the next data batch could change
            # their interpretation.
            # Find the last index that is safe too look at.
            last_i, _ = find_last_break(times=self.times['time'],
                                        last_time=self.last_time_searched,
                                        break_time=self.config['signal_separation'])
            if last_i == -1:
                # There are no breaks at all in the data! We need more data to continue.
                return
        else:
            # We can look at all added time.
            last_i = len(self.times) - 1

        # Select the times we can work with, save the rest for the next time we are called.
        # Note the + 1 for python's exclusive right index convention,
        # which still annoys me even after years of python programming.
        times = self.times[:last_i + 1]
        self.times = self.times[last_i + 1:]

        # Start the signal finder
        # does_channel_contribute is a buffer for internal use in signal_finder we can't allocate inside it
        # due to numba limitations
        does_channel_contribute = np.zeros(len(self.coincidence_tally), dtype=np.int)
        sigf = signal_finder(times=times,
                             signal_separation=self.config['signal_separation'],
                             signal_buffer=self.numba_signals_buffer,
                             coincidence_tally=self.coincidence_tally,
                             does_channel_contribute=does_channel_contribute,
                             dark_rate_save_interval=self.config['dark_rate_save_interval'])
        saved_buffers = []
        while True:
            result = next(sigf)
            if result >= 0:
                # All times processed!
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

        self.end_of_run_info['signals_found'] += len(signals)
        self.log.debug("Low-level trigger found %d signals in data." % len(signals))

        # Hand over to each of the high level trigger modules in turn.
        # Notice we won't even try to sort the event ranges from different triggers by time: some triggers (like the
        # main trigger) can hold events back until next time we give them data, so this would be a thorny problem.
        for hlt in self.hlts:
            yield from hlt.get_event_ranges(signals)

    def shutdown(self):
        """Shuts down trigger, return a dictionary with end-of-run information, for printing or for the runs database
        """
        self.save_dark_monitor_data(last_time=True)

        # Shutdown the high-level triggers
        for hlt in self.hlts:
            hlt.shutdown()

        # Close the trigger data file. Note this must be done after shutting down the hlts, they may add something
        # on shutdown as well.
        if hasattr(self, 'f'):
            self.f.close()

        # Add end-of-run info to the runs database
        self.end_of_run_info.update(dict(last_time_searched=self.last_time_searched,
                                         end_trigger_processing_timestamp=time.time()))
        for htl in self.hlts:
            self.end_of_run_info[hlt.name] = hlt.end_of_run_info

        return self.end_of_run_info


@numba.jit(nopython=True)
def get_pmt_numbers(channels, modules, pmts_buffer, pmt_lookup):
    """Fills pmts_buffer with pmt numbers corresponding to channels, modules accordint to pmt_lookup.
     - pmt_lookup: lookup matrix for pmt numbers. First index is digitizer module, second is digitizer channel.
    Modifies pmts_buffer in-place.
    """
    for i in range(len(channels)):
        pmts_buffer[i] = pmt_lookup[modules[i], channels[i]]


@numba.jit(nopython=True)
def signal_finder(times, signal_separation, signal_buffer, coincidence_tally,
                  does_channel_contribute,
                  dark_rate_save_interval):
    """Fill signal_buffer with signals in times. Other arguments:
     - signal_separation: group pulses into signals separated by signal_separation.
     - coincidence_tally: nxn matrix of zero where n is number of channels,used to store 2-pmt coincidences
       (with 1-pmt, i.e. dark rate, on diagonal)
     - dark_rate_save_interval: yield SAVE_DARK_MONITOR every dark_rate_save_interval
     - does channel contibute: zero array of len n_channels. Very annoying we can't allocate this inside!
    Online RMS algorithm is Knuth/Welford: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    in_signal = False
    passes_test = False     # Does the current time pass the signal inclusion test?
    current_signal = 0      # Index of the current signal in the signal buffer
    M2 = 0.0                # Temporary variable for online RMS computation
    if len(times):
        next_save_time = times[0].time + dark_rate_save_interval

    for time_index, time in enumerate(times):
        t = time.time
        pmt = time.pmt

        if t > next_save_time:
            yield SAVE_DARK_MONITOR_DATA
            next_save_time += dark_rate_save_interval

        # Should this time be in a signal -- is the next time close enough?
        is_last_time = time_index == len(times) - 1
        if not is_last_time:
            passes_test = times[time_index+1].time - t < signal_separation

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
def find_last_break(times, last_time, break_time):
    """Return the last index in times after which there is a gap >= signal_separation.
    If the last entry in times is further than signal_separation from last_time,
    that last index in times is returned.
    Returns -1 if no break exists anywhere in times.
    """
    for i, t in reversed(enumerate(times)):
        t = times[i]
        if t < last_time - break_time:
            return t
        else:
            last_time = t
    return -1


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
