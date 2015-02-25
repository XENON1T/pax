import numpy as np
import numba

# For diagnostic plotting:
import matplotlib.pyplot as plt
import os

from pax import plugin, datastructure, units


class FindHits(plugin.TransformPlugin):

    def startup(self):
        c = self.config
        
        self.min_sigma = c['peak_minimum_sigma']
        self.noise_sigma_guess = c['noise_sigma_guess']
        self.reference_baseline = c.get('digitizer_baseline', 16000)
        self.initial_baseline_samples = c.get('initial_baseline_samples', 50)
        self.max_hits_per_pulse = c['max_hits_per_pulse']
        self.max_passes = c.get('max_passes', float('inf'))

        self.make_diagnostic_plots = c.get('make_diagnostic_plots', 'never')
        self.make_diagnostic_plots_in = c.get('make_diagnostic_plots_in', 'small_pf_diagnostic_plots')
        if self.make_diagnostic_plots != 'never':
            if not os.path.exists(self.make_diagnostic_plots_in):
                os.makedirs(self.make_diagnostic_plots_in)

        # Conversion factor: multiply by this to convert from ADC counts above baseline -> electrons
        # Still has to be divided by PMT gain to go to photo-electrons (done below)
        self.conversion_factor = c['sample_duration'] * c['digitizer_voltage_range'] / (
            2 ** (c['digitizer_bits'])
            * c['pmt_circuit_load_resistor']
            * c['external_amplification']
            * units.electron_charge)


    def transform_event(self, event):
        # Keep count of number of pulses without hits in each channel
        noise_pulses_in = np.zeros(self.config['n_channels'], dtype=np.int)

        # Allocate numpy arrays to hold numba peakfinder results
        raw_peaks = -1 * np.ones((self.max_hits_per_pulse, 2), dtype=np.int64)
        argmaxes = np.zeros(self.max_hits_per_pulse, dtype=np.int64)
        areas = np.zeros(self.max_hits_per_pulse)

        for pulse_i, pulse in enumerate(event.occurrences):
            start = pulse.left
            stop = pulse.right
            channel = pulse.channel

            # Don't consider dead channels
            if self.config['gains'][channel] == 0:
                continue

            # Retrieve the waveform, subtract ref baseline, invert
            w = self.reference_baseline - pulse.raw_data.astype(np.float64)

            # Call the numba hit finder -- see its documentation below
            # Results stored in raw_peaks and extra_results; declared outside loop, see above
            n_raw_peaks_found, baseline, noise_sigma = self._find_peaks(
                w, self.min_sigma, self.noise_sigma_guess, self.max_passes, self.initial_baseline_samples,
                raw_peaks)

            if n_raw_peaks_found == len(raw_peaks) - 1:
                self.log.warning("Occurrence %s-%s in channel %s has more than %s hits!"
                                 "Probably the noise is unusually high. Further hits "
                                 "have been ignored." % (start, stop, channel,
                                                         self.max_hits_per_pulse))
                n_raw_peaks_found = len(raw_peaks)

            # Update the noise pulse count
            if n_raw_peaks_found == 0:
                noise_pulses_in[channel] += 1

            # Only view the part of raw_peaks that contains peaks found in this event
            # The rest of the array contains zeros orrandom stuff from previous events
            fresh_peaks = raw_peaks[:n_raw_peaks_found]

            # Update pulse data
            pulse.noise_sigma = noise_sigma
            pulse.baseline = baseline
            # TODO: compute and store pulse height

            # Compute area and max of each hit
            # Results stored in argmaxes, areas; declared outside loop, see above
            self._peak_argmax_and_area(w, fresh_peaks, argmaxes, areas)

            # Store the found peaks in the datastructure
            # Convert area and height from adc counts -> pe
            peaks = []
            conversion_factor = self.conversion_factor / self.config['gains'][channel]
            for i, p in enumerate(fresh_peaks):
                max_idx = p[0] + argmaxes[i]
                event.all_channel_peaks.append(datastructure.ChannelPeak({
                    'channel':             channel,
                    'left':                start + p[0],
                    'index_of_maximum':    start + max_idx,
                    'right':               start + p[1],
                    'area':                areas[i] * conversion_factor,
                    'height':              w[max_idx] * conversion_factor,
                    'noise_sigma':         noise_sigma,
                    'found_in_pulse':      pulse_i,
                }))

            # Diagnostic plotting
            # Can't more to plotting plugin: occurrence grouping of hits lost after clustering
            if self.make_diagnostic_plots == 'always' or \
               self.make_diagnostic_plots == 'no peaks' and not len(peaks):
                plt.figure()
                plt.plot(w, drawstyle='steps', label='data')
                for p in fresh_peaks:
                    plt.axvspan(p[0] - 1, p[1], color='red', alpha=0.5)
                plt.plot(noise_sigma * np.ones(len(w)), '--', label='1 sigma')
                plt.plot(self.min_sigma * noise_sigma * np.ones(len(w)),
                         '--', label='%s sigma' % self.min_sigma)
                # TODO: don't draw another line, draw another y-axis!
                plt.plot(np.ones(len(w)) * self.config['gains'][channel] / self.conversion_factor,
                         '--', label='1 pe/sample')
                plt.legend()
                bla = (event.event_number, start, stop, channel)
                plt.title('Event %s, occurrence %d-%d, Channel %d' % bla)
                plt.xlabel("Sample number (%s ns)" % event.sample_duration)
                plt.ylabel("Amplitude (ADC counts above baseline)")
                plt.savefig(os.path.join(self.make_diagnostic_plots_in,
                                         'event%04d_occ%05d-%05d_ch%03d.png' % bla))
                plt.close()

        # Mark channels with an abnormally high noise rate as bad
        bad_channels = np.where(noise_pulses_in > self.config['maximum_noise_occurrences_per_channel'])[0]
        event.is_channel_bad[bad_channels] = True

        return event

    # TODO: Needs a test!
    @staticmethod
    @numba.jit(numba.void(numba.float64[:], numba.int64[:, :], numba.int64[:], numba.float64[:]), nopython=True)
    def _peak_argmax_and_area(w, raw_peaks, argmaxes, areas):
        for peak_i in range(len(raw_peaks)):
            current_max = -999.9
            current_argmax = -1
            current_area = 0
            for i, x in enumerate(w[raw_peaks[peak_i, 0]:raw_peaks[peak_i, 1]+1]):
                if x > current_max:
                    current_max = x
                    current_argmax = i
                current_area += x
            argmaxes[peak_i] = current_argmax
            areas[peak_i] = current_area


    @staticmethod
    # Metaresults: baseline correction, noise level,
    @numba.jit(numba.typeof((1.0, 2.0, 3.0))(
                    numba.float64[:], numba.float64, numba.float64, numba.int64, numba.int64,
                    numba.int64[:, :]), nopython=True)
    def _find_peaks(w, threshold_sigmas, noise_sigma, max_passes, initial_baseline_samples, raw_peaks):
        """Fills raw_peaks with left, right indices of raw_peaks > bound_threshold which exceed threshold somewhere
         raw_peaks: numpy () of [-1,-1] lists, will be filled by function.
        Will modify w to do baseline correction
        Returns: number of raw_peaks found, baseline, noise_sigma
        Will stop search after raw_peaks found reached length of raw_peaks argument passed in
        """
        has_changed = True      # First pass is never final

        # Determine an initial baseline from the first samples
        # We have to do this BEFORE the first hit finding pass:
        # the entire pulse could be above reference baseline, and we don't want toreport it all as one hit...
        # TODO: Also determine a noise_sigma here? Factor out code from below to separate function?
        baseline = 0.0
        initial_baseline_samples = max(initial_baseline_samples, len(w))
        for x in w[:initial_baseline_samples]:
            baseline += x
        baseline /= initial_baseline_samples

        # Correct for the initial baseline
        for i in range(len(w)):
            w[i] -= baseline

        # Determine the peaks based on the noise level
        # Can't just use w > self.min_sigma * noise_sigma here,
        # want to extend peak bounds to noise_sigma
        pass_number = 0
        while True:

            in_candidate_interval = False
            current_interval_passed_test = False
            current_peak = 0
            max_peaks = len(raw_peaks) - 1
            max_idx = len(w) - 1
            current_candidate_interval_start = -1
            threshold = noise_sigma * threshold_sigmas
            bound_threshold = noise_sigma

            ##
            #   Hit finding
            ##

            for i, x in enumerate(w):

                if not in_candidate_interval and x > bound_threshold:
                    # Start of candidate interval
                    in_candidate_interval = True
                    current_candidate_interval_start = i

                # This must be if, not else: an interval can cross threshold in start sample
                if in_candidate_interval:

                    if x > threshold:
                        current_interval_passed_test = True

                    if x < bound_threshold or i == max_idx:
                        # End of candidate interval
                        in_candidate_interval = False

                        if current_interval_passed_test:
                            # We've found a new peak!

                            # Add to raw_peaks, check if something has changed
                            if i-1 != raw_peaks[current_peak, 1] \
                                    or current_candidate_interval_start != raw_peaks[current_peak, 0]:
                                raw_peaks[current_peak, 0] = current_candidate_interval_start
                                raw_peaks[current_peak, 1] = i - 1
                                has_changed = True

                            # Check if we've reached the maximum # of peaks
                            # If we found more peaks than we have room in our result array,
                            # stop peakfinding immediately
                            if current_peak == max_peaks:
                                in_candidate_interval = False
                                break

                            # Prepare for the next peak
                            current_peak += 1
                            current_interval_passed_test = False

            n_peaks_found = current_peak

            # If nothing has changed in peakfinding, no need to continue
            # If we have reached the max pass_number, we should stop here, not recalculate noise sigma again
            # (otherwise reported peaks will be inconsistent with reported baseline and noise)
            if not has_changed or pass_number > max_passes:
                break

            ##
            #   Noise sigma, baseline computation
            ##

            # Compute the baseline correction and the new noise_sigma
            # -- BuildWaveforms can get baseline wrong if there is a pe in the starting samples
            n = 0           # Samples outside peak included so far
            mean = 0        # Mean so far
            m2 = 0          # Sum of squares of differences from the (current) mean

            current_peak = 0
            currently_in_peak = False

            for i, x in enumerate(w):

                if currently_in_peak:
                    if i > raw_peaks[current_peak, 1]:
                        current_peak += 1
                        currently_in_peak = False

                # NOT else, currently_in_peak may have changed!
                if not currently_in_peak:
                    if current_peak < n_peaks_found and i == raw_peaks[current_peak, 0]:
                        currently_in_peak = True
                    else:
                        delta = x - mean
                        # Update n, mean and m2
                        n += 1
                        mean += delta/n
                        m2 += delta*(x-mean)

            noise_sigma = (m2/n)**0.5

            # Perform the baseline correction
            baseline += mean
            for i in range(len(w)):
                w[i] -= mean

            has_changed = False
            pass_number += 1

        # Return number of peaks found, put rest of info in extra_results
        # Convert n_peaks_found to float, if you keep it int, it will sometimes be int32, sometimes64
        return (float(n_peaks_found), baseline, noise_sigma)
