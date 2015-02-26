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

        self.build_sum_waveforms = c.get('build_sum_waveforms', False)

        self.make_diagnostic_plots = c.get('make_diagnostic_plots', 'never')
        self.make_diagnostic_plots_in = c.get('make_diagnostic_plots_in', 'small_pf_diagnostic_plots')
        if self.make_diagnostic_plots != 'never':
            if not os.path.exists(self.make_diagnostic_plots_in):
                os.makedirs(self.make_diagnostic_plots_in)

        # Conversion factor: multiply by this to convert from ADC counts above baseline -> electrons
        # Still has to be divided by PMT gain to go to photo-electrons (done below)
        self.adc_to_e = c['sample_duration'] * c['digitizer_voltage_range'] / (
            2 ** (c['digitizer_bits']) *
            c['pmt_circuit_load_resistor'] *
            c['external_amplification'] *
            units.electron_charge)

    def transform_event(self, event):
        # Allocate numpy arrays to hold numba peakfinder results
        # -42 is a placeholder for values that should never be displayed
        hits_buffer = -42 * np.ones((self.max_hits_per_pulse, 2), dtype=np.int64)
        argmaxes = -42 * np.zeros(self.max_hits_per_pulse, dtype=np.int64)
        areas = -42 * np.ones(self.max_hits_per_pulse)

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
            # Results stored in hits_buffer and extra_results; declared outside loop, see above
            n_hits_found, baseline, noise_sigma, passes_used = self._find_peaks(
                w, self.min_sigma, self.noise_sigma_guess, self.max_passes, self.initial_baseline_samples,
                hits_buffer)

            if n_hits_found >= self.max_hits_per_pulse:
                self.log.warning("Pulse %s-%s in channel %s has more than %s hits!"
                                 "This usually indicates a zero-length encoding breakdown after a very large S2."
                                 "Further hits have been ignored." % (start, stop, channel, self.max_hits_per_pulse))

            if passes_used >= self.max_passes:
                self.log.debug("Hit finding in pulse %d-%d in channel %d did not converge after %d passes." % (
                    start, stop, channel, passes_used))

            # If no hits were found, this is a noise pulse: update the noise pulse count
            if n_hits_found == 0:
                event.noise_pulses_in[channel] += 1
                # Don't 'continue' to the next pulse! There's stuff left to do!

            # Only view the part of hits_buffer that contains peaks found in this event
            # The rest of hits_buffer contains zeros or random junk from previous pulses
            hits_found = hits_buffer[:n_hits_found]

            # Update pulse data
            pulse.noise_sigma = noise_sigma
            pulse.baseline = baseline
            # TODO: compute and store pulse height

            # Compute area and max of each hit
            # Results stored in argmaxes, areas; declared outside loop, see above
            self._peak_argmax_and_area(w, hits_found, argmaxes, areas)

            # Store the found peaks in the datastructure
            # Convert area, noise_sigma and height from adc counts -> pe
            peaks = []
            adc_to_pe = self.adc_to_e / self.config['gains'][channel]
            for i, hit in enumerate(hits_found):

                # Do sanity checks
                area = areas[i] * adc_to_pe
                height = w[hit[0] + argmaxes[i]] * adc_to_pe
                left = start + hit[0]
                right = start + hit[1]
                max_idx = start + hit[0] + argmaxes[i]
                noise_sigma_pe = noise_sigma * adc_to_pe
                if not (0 <= left <= max_idx <= right) or not (0 <= self.min_sigma * noise_sigma_pe <= height <= area):
                    raise RuntimeError("You found a hitfinder bug!\n"
                                       "Current hit %d-%d-%d, in event %s, channel %s, pulse %s.\n"
                                       "Indices in pulse: %s-%s-%s. Pulse bounds: %d-%d.\n"
                                       "Height is %s, noise sigma is %s, dynamic threshold at %s; Area is %d.\n"
                                       "Please tell Jelle!" % (
                                           left, max_idx, right, event.event_number, channel, pulse_i,
                                           hit[0], hit[0] + argmaxes[i], hit[1], start, stop,
                                           height, noise_sigma_pe, self.min_sigma * noise_sigma_pe, area))

                event.all_channel_peaks.append(datastructure.ChannelPeak({
                    'channel':             channel,
                    'left':                left,
                    'index_of_maximum':    max_idx,
                    'right':               right,
                    'area':                area,
                    'height':              height,
                    'noise_sigma':         noise_sigma_pe,
                    'found_in_pulse':      pulse_i,
                }))

            # Diagnostic plotting
            # Can't more to plotting plugin: occurrence grouping of hits lost after clustering
            if self.make_diagnostic_plots == 'always' or \
               self.make_diagnostic_plots == 'no peaks' and not len(peaks):
                plt.figure()
                plt.plot(w, drawstyle='steps', label='data')
                for hit in hits_found:
                    plt.axvspan(hit[0] - 1, hit[1], color='red', alpha=0.5)
                plt.plot(noise_sigma * np.ones(len(w)), '--', label='1 sigma')
                plt.plot(self.min_sigma * noise_sigma * np.ones(len(w)),
                         '--', label='%s sigma' % self.min_sigma)
                # TODO: don't draw another line, draw another y-axis!
                plt.plot(np.ones(len(w)) * self.config['gains'][channel] / self.adc_to_e,
                         '--', label='1 pe/sample')
                plt.legend()
                bla = (event.event_number, start, stop, channel)
                plt.title('Event %s, occurrence %d-%d, Channel %d' % bla)
                plt.xlabel("Sample number (%s ns)" % event.sample_duration)
                plt.ylabel("Amplitude (ADC counts above baseline)")
                plt.savefig(os.path.join(self.make_diagnostic_plots_in,
                                         'event%04d_occ%05d-%05d_ch%03d.png' % bla))
                plt.close()

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
    @numba.jit(numba.typeof((1.0, 2.0, 3.0, 4.0))(
               numba.float64[:], numba.float64, numba.float64, numba.int64, numba.int64,
               numba.int64[:, :]), nopython=True)
    def _find_peaks(w, threshold_sigmas, noise_sigma, max_passes, initial_baseline_samples, hits_buffer):
        """Fills raw_peaks with left, right indices of raw_peaks > bound_threshold which exceed threshold somewhere
         raw_peaks: numpy () of [-1,-1] lists, will be filled by function.
        BE CAREFUL -- Will modify w IN PLACEto do baseline correction
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
        for x in w[:initial_baseline_samples + 1]:     # AARGH stupid python indexing!!!
            baseline += x
        baseline = baseline/initial_baseline_samples

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
            max_n_peaks = len(hits_buffer)      # First index which is outside hits buffer = max # of peaks to find
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

                            # The interval ended just before this index
                            # unless, of course, we ended ONLY BECAUSE this is the last index
                            itv_end = i-1 if x < bound_threshold else i

                            # Add to raw_peaks, check if something has changed
                            if itv_end != hits_buffer[current_peak, 1] \
                                    or current_candidate_interval_start != hits_buffer[current_peak, 0]:
                                hits_buffer[current_peak, 0] = current_candidate_interval_start
                                hits_buffer[current_peak, 1] = itv_end
                                has_changed = True

                            # Prepare for the next peak
                            current_peak += 1
                            current_interval_passed_test = False

                            # Check if we've reached the maximum # of peaks
                            # If we found more peaks than we have room in our result array,
                            # stop peakfinding immediately
                            if current_peak == max_n_peaks:
                                break

            n_peaks_found = current_peak

            # If we have reached the max pass_number, we should stop BEFORE recalculate noise sigma again
            # (otherwise reported peaks will be inconsistent with reported baseline and noise)
            # We can also stop if nothing has changed: baseline and noise will be the same as before
            if pass_number > max_passes or not has_changed:
                break

            ##
            #   Baseline computation
            ##

            # Compute the baseline correction
            n = 0           # Samples outside peak included so far
            mean = 0        # Mean so far

            current_peak = 0
            currently_in_peak = False

            for i, x in enumerate(w):

                if currently_in_peak:
                    if i > hits_buffer[current_peak, 1]:
                        current_peak += 1
                        currently_in_peak = False

                # NOT else, currently_in_peak may have changed!
                if not currently_in_peak:
                    if current_peak < n_peaks_found and i == hits_buffer[current_peak, 0]:
                        currently_in_peak = True
                    else:
                        # Update n, mean
                        n += 1
                        mean += (x - mean)/n

            # Perform the baseline correction
            baseline += mean
            for i in range(len(w)):
                w[i] -= mean

            ##
            #   Noise sigma computation
            ##

            # Must be in a second pass, since we want to include only samples BELOW baseline (which we just determined)
            m2 = 0          # Sum of squares of differences from the (current) mean
            n_negative = 0  # Samples below previously determined baseline so far

            current_peak = 0
            currently_in_peak = False

            for i, x in enumerate(w):

                if currently_in_peak:
                    if i > hits_buffer[current_peak, 1]:
                        current_peak += 1
                        currently_in_peak = False

                # NOT else, currently_in_peak may have changed!
                if not currently_in_peak:
                    if current_peak < n_peaks_found and i == hits_buffer[current_peak, 0]:
                        currently_in_peak = True
                    else:
                        if x < 0:
                            n_negative += 1
                            m2 += x*x

            noise_sigma = (m2/n_negative)**0.5
            has_changed = False
            pass_number += 1

        # Return number of peaks found, baseline, noise sigma, and number of passes used
        # Convert ints to float, if you keep it int, it will sometimes be int32, sometimes int64 => numba crashes
        return float(n_peaks_found), baseline, noise_sigma, float(pass_number + 1)
