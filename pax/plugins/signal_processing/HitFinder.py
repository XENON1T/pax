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

        self.initial_baseline_samples = c.get('initial_baseline_samples', 50)
        self.max_hits_per_pulse = c['max_hits_per_pulse']

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
        # Allocate numpy arrays to hold numba hitfinder results
        # -1 is a placeholder for values that should never appear (0 would be bad as it often IS a possible value)
        hits_buffer = -1 * np.ones((self.max_hits_per_pulse, 2), dtype=np.int64)
        argmaxes = -1 * np.ones(self.max_hits_per_pulse, dtype=np.int64)
        areas = -1 * np.ones(self.max_hits_per_pulse, dtype=np.float64)
        centers = -1 * np.ones(self.max_hits_per_pulse, dtype=np.float64)

        dt = self.config['sample_duration']

        for pulse_i, pulse in enumerate(event.pulses):
            start = pulse.left
            stop = pulse.right
            channel = pulse.channel
            pmt_gain = self.config['gains'][channel]

            # Retrieve waveform as floats: needed to subtract baseline (which can be in between ADC counts)
            w = pulse.raw_data.astype(np.float64)

            # Subtract reference baseline, invert (so hits point up from baseline)
            # This is convenient so we don't have to reinterpret min, max, etc
            w = self.config['digitizer_reference_baseline'] - w

            pulse.baseline, pulse.noise_sigma, pulse.minimum, pulse.maximum = \
                compute_pulse_properties(w, self.initial_baseline_samples)

            w -= pulse.baseline

            # Don't do hitfinding in dead channels, pulse property computation was enough
            # Could refactor pulse property computation to separate plugin,
            # but that would mean waveform has to be converted to floats twice
            if pmt_gain == 0:
                continue

            # Compute thresholds based on noise level
            high_threshold = max(self.config['height_over_noise_high_threshold'] * pulse.noise_sigma,
                                 self.config['absolute_adc_counts_high_threshold'],
                                 - self.config['height_over_min_high_threshold'] * pulse.minimum)
            low_threshold = max(self.config['height_over_noise_low_threshold'] * pulse.noise_sigma,
                                self.config['absolute_adc_counts_low_threshold'],
                                - self.config['height_over_min_low_threshold'] * pulse.minimum)

            # Call the numba hit finder -- see its docstring for description
            n_hits_found = find_intervals_above_threshold(w, high_threshold, low_threshold, hits_buffer)

            # Only view the part of hits_buffer that contains hits found in this event
            # The rest of hits_buffer contains -1's or stuff from previous pulses
            hits_found = hits_buffer[:n_hits_found]

            # If no hits were found, this is a noise pulse: update the noise pulse count
            if n_hits_found == 0:
                event.noise_pulses_in[channel] += 1
                # Don't 'continue' to the next pulse! There's stuff left to do!
            # Show too-many-hits message if needed
            # This message really should be shown the first few times, as you should be aware how often this occurs
            elif n_hits_found >= self.max_hits_per_pulse:
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

            # Compute area, max, and center of each hit in numba
            # Results stored in argmaxes, areas, centers; declared outside loop, see above
            compute_hit_properties(w, hits_found, argmaxes, areas, centers)

            # Store the found hits in the datastructure
            # Convert area, noise_sigma and height from adc counts -> pe
            adc_to_pe = self.adc_to_e / pmt_gain
            noise_sigma_pe = pulse.noise_sigma * adc_to_pe
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
                   or not (0 <= high_threshold * adc_to_pe <= height <= area):
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
                                           height, noise_sigma_pe, high_threshold * adc_to_pe,
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
            # Difficult to move to separate plugin:
            # would have to re-group hits by pulse, move settings like threshold to DEFAULT or use hack...

            # Do we need to show this pulse? If not: continue
            if self.make_diagnostic_plots == 'never':
                continue
            elif self.make_diagnostic_plots == 'tricky cases':
                # Always show pulse if noise level is very high
                if noise_sigma_pe < 0.5:
                    if len(hits_found) == 0:
                        # Show pulse if it nearly went over threshold
                        if not pulse.maximum > 0.8 * high_threshold:
                            continue
                    else:
                        # Show pulse if any of its hit nearly didn't go over threshold
                        if not any([event.all_hits[-(i+1)].height < 1.2 * high_threshold * adc_to_pe
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
            ax1.plot(np.ones_like(w) * high_threshold, '--', label='Threshold', color='red')
            ax1.plot(np.ones_like(w) * pulse.noise_sigma, '--', label='Noise level', color='gray')
            ax1.plot(np.ones_like(w) * low_threshold, '--', label='Boundary threshold', color='green')

            # Mark the hit ranges & center of gravity point
            for hit_i, hit in enumerate(hits_found):
                ax1.axvspan(hit[0] - 0.5, hit[1] + 0.5, color='red', alpha=0.2)
                # Remember: array 'centers' is still in samples since start of hit...
                ax1.axvline([centers[hit_i] + hit[0]], linestyle=':', color='gray')

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


@numba.jit(nopython=True)
def find_intervals_above_threshold(w, high_threshold, low_threshold, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w > low_threshold which exceed high_threshold somewhere
        result_buffer: numpy N*2 array of ints, will be filled by function.
    Returns: number of intervals found
    Will stop search after N intervals are found, with N the length of result_buffer.
    Boundary indices are inclusive, i.e. the right index is the last index which was still above low_threshold
    """
    # This function just wraps the numba code below, so you can call it with keyword arguments
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

            if x <= low_threshold or i == last_index_in_w:

                # End of candidate interval
                in_candidate_interval = False

                if current_interval_passed_test:
                    # We've found a new interval!

                    # The interval ended just before this index
                    # unless, of course, we ended ONLY BECAUSE this is the last index
                    itv_end = i-1 if x <= low_threshold else i

                    # Add to result buffer
                    result_buffer[current_interval, 0] = current_candidate_interval_start
                    result_buffer[current_interval, 1] = itv_end

                    # Prepare for the next interval
                    current_interval += 1
                    current_interval_passed_test = False

                    if current_interval == result_buffer_size:
                        break

    # Return number of hits found
    # One day numba may have crashed here: not sure if it is int32 or int64...
    return current_interval


@numba.jit(nopython=True)
def compute_hit_properties(w, raw_hits, argmaxes, areas, centers):
    """Finds the maximum index, area, and center of gravity of hits in w indicated by (l, r) bounds in raw_hits.
    Will fill up argmaxes and areas with result.
    raw_hits should be a numpy array of (left, right) bounds (inclusive)
    centers, argmaxes are returned in samples right of hit start -- you probably want to convert this
    Returns nothing
    """
    for hit_i in range(len(raw_hits)):
        current_max = -999.9
        current_argmax = -1
        current_area = 0.0
        current_center = 0.0
        for i, x in enumerate(w[raw_hits[hit_i, 0]:raw_hits[hit_i, 1]+1]):
            if x > current_max:
                current_max = x
                current_argmax = i
            current_area += x
            current_center += i * x
        argmaxes[hit_i] = current_argmax
        areas[hit_i] = current_area
        centers[hit_i] = current_center / current_area


@numba.jit(nopython=True)
def compute_pulse_properties(w, initial_baseline_samples):
    """Compute basic pulse properties quickly
    :param w: Raw pulse waveform in ADC counts
    :param initial_baseline_samples: number of samples at start of pulse to use for baseline computation
    :return: (baseline, noise_sigma, min, max);
      min and max are relative to baseline
      noise_sigma is the std of samples below baseline
    Does not modify w. Does not assume anything about inversion of w!!
    """

    # First compute baseline
    baseline = 0.0
    initial_baseline_samples = min(initial_baseline_samples, len(w))
    for x in w[:initial_baseline_samples]:
        baseline += x
    baseline /= initial_baseline_samples

    # Now compute mean, noise, and min
    n = 0           # Running count of samples included in noise sample
    m2 = 0          # Running sum of squares of differences from the baseline
    max_a = -1.0e6  # Running max amplitude
    min_a = 1.0e6   # Running min amplitude

    for x in w:
        if x > max_a:
            max_a = x
        if x < min_a:
            min_a = x
        if x < baseline:
            delta = x - baseline
            n += 1
            m2 += delta*(x-baseline)

    if n == 0:
        # Should only happen if w = baseline everywhere
        noise = 0
    else:
        noise = (m2/n)**0.5

    return baseline, noise, min_a - baseline, max_a - baseline
