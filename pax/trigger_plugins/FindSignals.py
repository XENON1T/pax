import numpy as np
import numba
from pax.trigger import TriggerPlugin, h5py_append
from pax.datastructure import TriggerSignal
from pax.dsputils import adc_to_pe

# Interrupts thrown by the signal finder
# Negative, since positive numbers indicate number of signals found during normal operation
SIGNAL_BUFFER_FULL = -1
SAVE_DARK_MONITOR_DATA = -2


class FindSignals(TriggerPlugin):

    # How often did we save the dark rate since the last full (coincidence matrix) save?
    dark_monitor_saves = 0

    def startup(self):
        # Initialize buffer for numba signal finding routine.
        # Although we're able to extend this buffer as needed, we must do so outside numba
        # and it involves copyping data, so if you pick too small a buffer size you will hurt performance.
        self.numba_signals_buffer = np.zeros(self.config['numba_signal_buffer_size'],
                                             dtype=TriggerSignal.get_dtype())

        # Initialize buffers for tallying pulses / coincidences
        n_channels = len(self.pmt_data) + 1     # Reason for +1 is again 'ghost' channels, see trigger.py
        self.all_pulses_tally = np.zeros(n_channels, dtype=np.int)
        self.lone_pulses_tally = np.zeros(n_channels, dtype=np.int)
        self.coincidence_tally = np.zeros((n_channels, n_channels), dtype=np.int)

        # Get conversion factor from ADC counts to pe for each pmt
        # The 'ghost' PMT will have gain 1 always
        self.gain_conversion_factors = np.array([adc_to_pe(self.trigger.pax_config['DEFAULT'], ch)
                                                 for ch in range(n_channels - 1)] +
                                                [1])

        # Create the dark monitoring datasets
        f = self.trigger.dark_monitor_data_file
        self.lone_pulses_dataset = f.create_dataset('lone_pulses',
                                                    shape=(0, n_channels),
                                                    maxshape=(None, n_channels),
                                                    dtype=np.int,
                                                    compression="gzip")
        self.lone_pulses_dataset.attrs['save_interval'] = self.config['dark_rate_save_interval']

        self.all_pulses_dataset = f.create_dataset('all_pulses',
                                                   shape=(0, n_channels),
                                                   maxshape=(None, n_channels),
                                                   dtype=np.int,
                                                   compression="gzip")
        self.all_pulses_dataset.attrs['save_interval'] = self.config['dark_rate_save_interval']

        self.coincidence_rate_dataset = f.create_dataset('coincidence_tally',
                                                         shape=(0, n_channels, n_channels),
                                                         maxshape=(None, n_channels, n_channels),
                                                         dtype=np.int,
                                                         compression="gzip")
        self.coincidence_rate_dataset.attrs['save_interval'] = (self.config['dark_rate_save_interval'] *
                                                                self.config['dark_monitor_full_save_every'])

        # We must keep track of the next time to save the dark rate between batches, since a batch usually does not
        # end exactly at a save time.
        self.next_save_time = None

    def process(self, data):
        if self.next_save_time is None:
            self.next_save_time = self.config['dark_rate_save_interval']
            if len(data.times):
                self.next_save_time += data.times['time'][0]

        sigf = signal_finder(times=data.times,
                             signal_separation=self.config['signal_separation'],

                             signal_buffer=self.numba_signals_buffer,

                             next_save_time=self.next_save_time,
                             dark_rate_save_interval=self.config['dark_rate_save_interval'],

                             all_pulses_tally=self.all_pulses_tally,
                             lone_pulses_tally=self.lone_pulses_tally,
                             coincidence_tally=self.coincidence_tally,

                             gain_conversion_factors=self.gain_conversion_factors,
                             )
        saved_buffers = []
        for result in sigf:

            if result >= 0:
                n_signals_found = result
                if len(saved_buffers):
                    self.log.debug("%d previous signal buffers were saved, concatenating and returning them." % (
                        len(saved_buffers)))
                    saved_buffers.append(self.numba_signals_buffer[:n_signals_found])
                    signals = np.concatenate(saved_buffers)
                else:
                    signals = self.numba_signals_buffer[:n_signals_found]
                break

            elif result == SIGNAL_BUFFER_FULL:
                self.log.debug("Signal buffer is full, copying it out.")
                saved_buffers.append(self.numba_signals_buffer.copy())

            elif result == SAVE_DARK_MONITOR_DATA:
                self.save_dark_monitor_data()
                self.next_save_time += self.config['dark_rate_save_interval']

            else:
                raise ValueError("Unknown signal finder interrupt %d!" % result)

        if data.last_data:
            self.save_dark_monitor_data(last_time=True)

        self.log.debug("Signal finder finished on this data increment, found %d signals." % len(signals))
        data.signals = signals

    def save_dark_monitor_data(self, last_time=False):

        # Save the PMT dark rate
        self.log.debug("Saving pulse rate: %d pulses (of which %d lone pulses)" % (
            self.all_pulses_tally.sum(), self.lone_pulses_tally.sum()))
        h5py_append(self.all_pulses_dataset, self.all_pulses_tally)
        self.all_pulses_tally *= 0
        h5py_append(self.lone_pulses_dataset, self.lone_pulses_tally)
        self.lone_pulses_tally *= 0

        self.dark_monitor_saves += 1

        if last_time or self.dark_monitor_saves == self.config['dark_monitor_full_save_every']:
            # Save the full coincidence rate
            self.log.debug("Saving coincidence tally matrix, total %d" % self.coincidence_tally.sum())
            h5py_append(self.coincidence_rate_dataset, self.coincidence_tally)
            self.dark_monitor_saves = 0
            self.coincidence_tally *= 0


def signal_finder(times, signal_separation,
                  signal_buffer,
                  next_save_time, dark_rate_save_interval,
                  all_pulses_tally, lone_pulses_tally, coincidence_tally,
                  gain_conversion_factors):
    """Fill signal_buffer with signals in times. Other arguments:
     - signal_separation: group pulses into signals separated by signal_separation.
     - coincidence_tally: nxn matrix of zero where n is number of channels,used to store 2-pmt coincidences
       (with 1-pmt, i.e. dark rate, on diagonal)
     - next_save_time: next time (in ns since start of run) the dark rate should be saved
     - dark_rate_save_interval: yield SAVE_DARK_MONITOR every dark_rate_save_interval
    Raises "interrupts" (yield numbers) to communicate with caller.
    Online RMS algorithm is Knuth/Welford: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    # Allocate memory for some internal buffers (which we can't do in numba) we don't need outside the signal finder
    n_channels = len(all_pulses_tally)   # Actually this is 1 more than the number of connected channels, see above
    does_channel_contribute = np.zeros(n_channels, dtype=np.int8)   # Bool gives weird errors
    area_per_channel = np.zeros(n_channels, dtype=np.float64)
    return _signal_finder(times, signal_separation,
                          signal_buffer,
                          next_save_time, dark_rate_save_interval,
                          all_pulses_tally, lone_pulses_tally, coincidence_tally,
                          gain_conversion_factors,
                          area_per_channel, does_channel_contribute)


@numba.jit()
def _signal_finder(times, signal_separation,
                   signal_buffer,
                   next_save_time, dark_rate_save_interval,
                   all_pulses_tally, lone_pulses_tally, coincidence_tally,
                   gain_conversion_factors,
                   area_per_channel, does_channel_contribute):
    """Numba backend for signal_finder: please see its docstring instead."""
    in_signal = False
    passes_test = False     # Does the current time pass the signal inclusion test?
    current_signal = 0      # Index of the current signal in the signal buffer
    m2 = 0.0                # Temporary variable for online RMS computation
    if not len(times):
        yield 0             # no point looking for events. Communicate no events found, then exit.
        return

    for time_index, _time in enumerate(times):
        t = _time.time
        pmt = _time.pmt
        area = _time.area * gain_conversion_factors[pmt]

        if t >= next_save_time:
            yield SAVE_DARK_MONITOR_DATA
            next_save_time += dark_rate_save_interval

        is_last_time = time_index == len(times) - 1
        if not is_last_time:
            # Should this time be in a signal? === Is the next time close enough?
            passes_test = times[time_index+1].time - t < signal_separation

        if not in_signal and passes_test:
            # Start a signal. We must clear all attributes first to remove (potential) old stuff from the buffer.
            in_signal = True
            s = signal_buffer[current_signal]
            s.left_time = t
            s.right_time = 0
            s.time_mean = 0
            s.time_rms = 0
            s.n_pulses = 0
            s.n_contributing_channels = 0
            s.area = 0
            area_per_channel *= 0
            does_channel_contribute *= 0

        if in_signal:                           # Notice if, not elif. Work on first time in signal too.
            # Update signal quantities
            s = signal_buffer[current_signal]
            area_per_channel[pmt] += area
            does_channel_contribute[pmt] = True
            s.n_pulses += 1
            delta = t - s.time_mean
            s.time_mean += delta / s.n_pulses
            m2 += delta * (t - s.time_mean)     # Notice this isn't delta**2: time_mean changed on the previous line!

            if not passes_test or is_last_time:
                # Signal has ended: store its quantities and move on
                s.right_time = t
                s.time_rms = (m2 / s.n_pulses)**0.5
                s.n_contributing_channels = does_channel_contribute.sum()
                s.area = area_per_channel.sum()
                if s.n_contributing_channels == 2:
                    indices = np.nonzero(does_channel_contribute)[0]
                    coincidence_tally[indices[0], indices[1]] += 1

                current_signal += 1
                m2 = 0
                in_signal = False

                if current_signal == len(signal_buffer):
                    yield SIGNAL_BUFFER_FULL
                    # Caller will have copied out the signal buffer, we can start from 0 again
                    current_signal = 0

        else:
            lone_pulses_tally[pmt] += 1

        all_pulses_tally[pmt] += 1

    # Let caller know number of signals found, then raise StopIteration
    yield current_signal
