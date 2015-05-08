"""Hit finding plugin

If you get an error from either of the numba methods in this plugin (exception from native function blahblah)
Try commenting the @jit decorators, which will run a slow, pure-python version of the methods, allowing you to debug.
Don't forget to re-enable the @jit -- otherwise it will run quite slow!
"""


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
        self.reference_baseline = c.get('digitizer_reference_baseline', 16000)
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

        # Keep track of how many times the "too many hits" warning has been shown
        self.too_many_hits_warnings_shown = 0

    def transform_event(self, event):
        # Allocate numpy arrays to hold numba peakfinder results
        # -1 is a placeholder for values that should never be used
        hits_buffer = -1 * np.ones((self.max_hits_per_pulse, 2), dtype=np.int64)
        argmaxes = -1 * np.zeros(self.max_hits_per_pulse, dtype=np.int64)
        areas = -1 * np.ones(self.max_hits_per_pulse, dtype=np.float64)
        centers = -1 * np.ones(self.max_hits_per_pulse, dtype=np.float64)

        dt = self.config['sample_duration']

        for pulse_i, pulse in enumerate(event.pulses):
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

            # Check for error return codes from the numba hitfinding
            if n_hits_found < 0:
                raise RuntimeError("You found a hitfinder bug!\n"
                                   "Event %d, channel %d, pulse %d.\n"
                                   "Guru meditation: %d\n"
                                   "Please file a bug report!" % (
                                       event.event_number, channel, pulse_i, n_hits_found))

            # Show too-many hits message
            if n_hits_found >= self.max_hits_per_pulse:
                if self.too_many_hits_warnings_shown > 3:
                    show_to = self.log.debug
                else:
                    show_to = self.log.info
                show_to("Pulse %s-%s in channel %s has more than %s hits. "
                        "This usually indicates a zero-length encoding breakdown after a very large S2. "
                        "Further hits in this pulse have been ignored." % (start, stop, channel,
                                                                           self.max_hits_per_pulse))
                self.too_many_hits_warnings_shown += 1
                if self.too_many_hits_warnings_shown == 3:
                    self.log.info('Further too-many hit messages will be suppressed!')

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
            pulse.height = np.max(w)    # Remember w was modified in-place by the numba hitfinder to do baselining

            # Compute area, max, and center of each hit in numba
            # Results stored in argmaxes, areas, centers; declared outside loop, see above
            self._compute_hit_properties(w, hits_found, argmaxes, areas, centers)

            # Store the found peaks in the datastructure
            # Convert area, noise_sigma and height from adc counts -> pe
            adc_to_pe = self.adc_to_e / self.config['gains'][channel]
            noise_sigma_pe = noise_sigma * adc_to_pe
            for i, hit in enumerate(hits_found):

                # Do sanity checks
                area = areas[i] * adc_to_pe
                center = (centers[i] + start + hit[0]) * dt
                height = w[hit[0] + argmaxes[i]] * adc_to_pe
                left = start + hit[0]
                right = start + hit[1]
                max_idx = start + hit[0] + argmaxes[i]
                if not (0 <= left <= max_idx <= right) \
                   or not (left <= center / dt <= right) \
                   or not (0 <= self.min_sigma * noise_sigma_pe <= height <= area):
                    raise RuntimeError("You found a hitfinder bug!\n"
                                       "Current hit %d-%d-%d, in event %s, channel %s, pulse %s.\n"
                                       "Indices in pulse: %s-%s-%s. Pulse bounds: %d-%d.\n"
                                       "Center of gravity at %s.\n"
                                       "Height is %s, noise sigma is %s, threshold at %s.\n"
                                       "Area is %d.\n"
                                       "Please file a bug report!" % (
                                           left, max_idx, right, event.event_number, channel, pulse_i,
                                           hit[0], hit[0] + argmaxes[i], hit[1], start, stop,
                                           center,
                                           height, noise_sigma_pe, self.min_sigma * noise_sigma_pe,
                                           area))

                event.all_hits.append(datastructure.Hit({
                    'channel':             channel,
                    'left':                left,
                    'index_of_maximum':    max_idx,
                    'center':              center,
                    'right':               right,
                    'area':                area,
                    'height':              height,
                    'noise_sigma':         noise_sigma_pe,
                    'found_in_pulse':      pulse_i,
                }))

            # Diagnostic plotting
            # Bit difficult to move to separate plugin: would have to re-group hits by pulse

            # Do we need to show this pulse? If not: continue
            if self.make_diagnostic_plots == 'never':
                continue
            elif self.make_diagnostic_plots == 'tricky cases':
                # Always show pulse if noise level is very high
                if noise_sigma_pe < 0.5:
                    if len(hits_found) == 0:
                        # Show pulse if it nearly went over threshold
                        if not pulse.height / noise_sigma > 0.8 * self.min_sigma:
                            continue
                    else:
                        # Show pulse if any of its hit nearly didn't go over threshold
                        if not any([event.all_hits[-(i+1)].height < 1.2 * self.min_sigma * noise_sigma_pe
                                   for i in range(len(hits_found))]):
                            continue
            elif self.make_diagnostic_plots == 'no hits':
                if len(hits_found) != 0:
                    continue
            elif self.make_diagnostic_plots == 'hits only':
                if len(hits_found) == 0:
                    continue
            else:
                if self.make_diagnostic_plots != 'always':
                    raise ValueError("Invalid make_diagnostic_plots option: %s!" % self.make_diagnostic_plots)

            # Setup the twin-y-axis plot
            fig, ax1 = plt.subplots(figsize=(10, 7))
            ax2 = ax1.twinx()
            ax1.set_xlabel("Sample number (%s ns)" % event.sample_duration)
            ax1.set_ylabel("ADC counts above baseline")
            ax2.set_ylabel("pe / sample")

            # Plot the signal and noise levels
            ax1.plot(w, drawstyle='steps-mid', label='Data')
            ax1.plot(np.ones_like(w) * self.min_sigma * noise_sigma, '--', label='Threshold', color='red')
            ax1.plot(np.ones_like(w) * noise_sigma, '--', label='Noise level', color='gray')
            ax1.plot(np.ones_like(w) * 2 * noise_sigma, '--', label='Boundary threshold', color='green')

            # Mark the hit ranges & center of gravity point
            for hit_i, hit in enumerate(hits_found):
                ax1.axvspan(hit[0] - 0.5, hit[1] + 0.5, color='red', alpha=0.2)
                ax1.axvline([centers[i] + hit[0]], linestyle=':', color='gray')

            # Make sure the y-scales match
            ax2.set_ylim(ax1.get_ylim()[0] * adc_to_pe, ax1.get_ylim()[1] * adc_to_pe)

            # Finish the plot, save, close
            leg = ax1.legend()
            leg.get_frame().set_alpha(0.5)
            bla = (event.event_number, start, stop, channel)
            plt.title('Event %s, pulse %d-%d, Channel %d' % bla)
            plt.savefig(os.path.join(self.make_diagnostic_plots_in,
                                     'event%04d_pulse%05d-%05d_ch%03d.png' % bla))
            plt.close()

        return event

    # TODO: Needs a test!
    @staticmethod
    @numba.jit(numba.void(numba.float64[:], numba.int64[:, :], numba.int64[:], numba.float64[:], numba.float64[:]),
               nopython=True)
    def _compute_hit_properties(w, raw_peaks, argmaxes, areas, centers):
        """Finds the maximum index, area, and center of gravity of hits in w indicated by (l, r) bounds in raw_peaks.
        Will fill up argmaxes and areas with result.
        raw_peaks should be a numpy array of (left, right) bounds (inclusive)
        centers, argmaxes are returned in samples right of hit start -- you probably want to convert this
        Returns nothing
        """
        for peak_i in range(len(raw_peaks)):
            current_max = -999.9
            current_argmax = -1
            current_area = 0
            current_center = 0
            for i, x in enumerate(w[raw_peaks[peak_i, 0]:raw_peaks[peak_i, 1]+1]):
                if x > current_max:
                    current_max = x
                    current_argmax = i
                current_area += x
                current_center += i * x
            argmaxes[peak_i] = current_argmax
            areas[peak_i] = current_area
            centers[peak_i] = current_center / current_area

    @staticmethod
    @numba.jit(numba.typeof((1.0, 2.0, 3.0, 4.0))(
               numba.float64[:], numba.float64, numba.float64, numba.int64, numba.int64,
               numba.int64[:, :]), nopython=True)
    def _find_peaks(w, threshold_sigmas, noise_sigma, max_passes, initial_baseline_samples, hits_buffer):
        """Fills raw_peaks with left, right indices of raw_peaks > 2 * noise which exceed threshold somewhere
         raw_peaks: numpy () of [-1,-1] lists, will be filled by function.
        BE CAREFUL -- Will modify w IN PLACE to do baseline correction
        Returns: number of raw_peaks found, baseline, noise_sigma
        Will stop search after raw_peaks found reached length of raw_peaks argument passed in
        """
        has_changed = True      # First pass is never final

        # Determine an initial baseline from the first samples
        # We have to do this BEFORE the first hit finding pass:
        # the entire pulse could be above reference baseline, and we don't want to report it all as one hit...
        # TODO: Also determine a noise_sigma here? Factor out code from below to separate function?
        baseline = 0.0
        initial_baseline_samples = min(initial_baseline_samples, len(w))
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
            bound_threshold = 2 * noise_sigma

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
            # Don't need to keep track of being in a peak or not: peaks are never below baseline :-)
            m2 = 0             # Sum of squares of differences from the (current) mean
            n_nonpositive = 0  # Number of samples <= baseline. <=, not < to avoid div0 error if no noise (if only..)

            for x in w:
                if x <= 0:
                    n_nonpositive += 1
                    m2 += x*x

            if n_nonpositive == 0:
                # A pathological case when the entire pulse is indicated as a peak (for this pass)...
                # I believe this can only happen if the baseline correction is bugged, so I'll raise an exception
                # ... Wait,Numba doesn't allow this though, so...
                return float(-41), 0.0, 0.0, 0.0

            noise_sigma = (m2/n_nonpositive)**0.5
            has_changed = False
            pass_number += 1

        # Return number of peaks found, baseline, noise sigma, and number of passes used
        # Convert ints to float, if you keep it int, it will sometimes be int32, sometimes int64 => numba crashes
        return float(n_peaks_found), baseline, noise_sigma, float(pass_number + 1)


# TODO: replace monstrous code above by nice code below
# TODO: And add back tests...
# TODO: Wait, int16 vs float64 waveforms.... aargh...
@numba.jit((numba.int64)(numba.int16[:], numba.float64, numba.float64, numba.int64[:, :]), nopython=True)
def find_intervals_above_threshold(w, high_threshold, low_threshold, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w > low_threshold which exceed high_threshold somewhere
        result_buffer: numpy N*2 array of ints, will be filled by function. 
    Returns: number of intervals found
    Will stop search after raw_peaks found reached N (length of raw_peaks argument passed in).
    Boundary indices are inclusive, i.e. the right index is the last index which was still above low_threshold
    """
    in_candidate_interval = False
    current_interval_passed_test = False
    current_interval = 0
    result_buffer_size = len(result_buffer)
    last_index_in_w = len(w) - 1
    current_candidate_interval_start = -1

    for i, x in enumerate(w):

        if not in_candidate_interval and x > low_threshold:
            # Start of candidate interval
            in_candidate_interval = True
            current_candidate_interval_start = i

        # This must be if, not else: an interval can cross high_threshold in start sample
        if in_candidate_interval:

            if x > high_threshold:
                current_interval_passed_test = True

            if x < low_threshold or i == last_index_in_w:

                # End of candidate interval
                in_candidate_interval = False

                if current_interval_passed_test:
                    # We've found a new interval!

                    # The interval ended just before this index
                    # unless, of course, we ended ONLY BECAUSE this is the last index
                    itv_end = i-1 if x < low_threshold else i

                    # Add to result buffer
                    result_buffer[current_interval, 0] = current_candidate_interval_start
                    result_buffer[current_interval, 1] = itv_end

                    # Prepare for the next interval
                    current_interval += 1
                    current_interval_passed_test = False

                    if current_interval == result_buffer_size:
                        break

    # Return number of peaks found
    # May crash numba here: not sure if it is int32 or int64...
    return current_interval