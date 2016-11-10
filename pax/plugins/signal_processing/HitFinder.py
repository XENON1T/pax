import numpy as np
import numba

# For diagnostic plotting:
from textwrap import dedent
import matplotlib.pyplot as plt
import os

from pax import plugin, datastructure, dsputils
from pax.dsputils import find_intervals_above_threshold


class FindHits(plugin.TransformPlugin):
    """Finds hits in pulses
    First, the waveform is baseline-corrected (with baseline as determined by PulseProperties.PulseProperties).
    Next, hits are found based on a high and low threshold:
     - A hit starts/ends when it passes the low_threshold
     - A hit has to pass the high_threshold somewhere.

    Thresholds can be set on three different quantities. These are computed per pulse, the highest is used.
    Here high/positive means 'signal like' (i.e. the PMT voltage becomes more negative)

    1) Height over noise threshold
           Options: height_over_noise_high_threshold, height_over_noise_low_threshold
       This threshold operates on the height above baseline / noise level
       The noise level is deteremined in each pulse as
         (<(w - baseline)**2>)**0.5
       with the average running *only* over samples < baseline!

    2) Absolute ADC counts above baseline
           Options: absolute_adc_counts_high_threshold, absolute_adc_counts_low_threshold
       If there is measurable noise in the waveform, you don't want the thresholds to fall to 0 and crash pax.
       This happens for pulses constant on initial_baseline_samples.
       Please use this as a deep fallback only, unless you know what mess you're getting yourself into!

    3) - Height / minimum
           Options: height_over_min_high_threshold, height_over_min_LOW_threshold
       Using this threshold protects you against sudden up & down fluctuations, especially in large pulses
       However, keep in mind that one large downfluctuation will cause a sensitivity loss throughout an entire pulse;
       if you have ZLE off, this may be a bad idea...
       This threshold operates on the (-) height above baseline / height below baseline of the lowest sample in pulse


    Edge cases:

    1) After large peaks the zero-length encoding can fail, making a huge pulse with many hits.
       If more than max_hits_per_pulse hits are found in a pulse, the rest will be ignored.
       If this threshold is set too low, you risk missing some hits in such events.
       If set too high, it will degrade performance. Don't set to infinity, we need to allocate memory for this...

    2) Very high hits (near ADC saturation) sometimes have a long tail. To protect against this, we raise the
       low threshold to a fraction dynamic_low_threshold_coeff of the hit height after a hit has been encountered.
       This is a temporary change for just the remainder of a pulse.
       Also, if the hit height * dynamic_low_threshold_coeff is lower than the low threshold, nothing is changed.

    Diagnostic plot options:
        make_diagnostic_plots can be always, never, tricky cases, no hits, hits only. This controls whether to make
        diagnostic plots showing individual pulses and the hitfinder's interpretation of them. For details on what
        constitutes a tricky case, check the source.
        make_diagnostic_plots_in sets the directory where the diagnostic plots are created.

    Debugging tip:
    If you get an error from one of the numba methods in this plugin (exception from native function blahblah)
    Try commenting the @jit decorators, which will run a slow, pure-python version of the methods,
    allowing you to debug. Don't forget to re-enable the @jit -- otherwise it will run quite slow!
    """

    def startup(self):
        c = self.config

        self.max_hits_per_pulse = c['max_hits_per_pulse']
        self.reference_baseline = self.config['digitizer_reference_baseline']

        self.make_diagnostic_plots = c.get('make_diagnostic_plots', 'never')
        self.make_diagnostic_plots_in = c.get('make_diagnostic_plots_in', 'small_pf_diagnostic_plots')

        if self.make_diagnostic_plots != 'never':
            if not os.path.exists(self.make_diagnostic_plots_in):
                os.makedirs(self.make_diagnostic_plots_in)

    def transform_event(self, event):
        dt = self.config['sample_duration']
        hits_per_pulse = []

        dynamic_low_threshold_coeff = self.config['dynamic_low_threshold_coeff']

        # Allocate numpy arrays to hold numba hitfinder results
        # -1 is a placeholder for values that should never appear (0 would be bad as it often IS a possible value)
        hit_bounds_buffer = -1 * np.ones((self.max_hits_per_pulse, 2), dtype=np.int64)
        hits_buffer = np.zeros(self.max_hits_per_pulse, dtype=datastructure.Hit.get_dtype())

        for pulse_i, pulse in enumerate(event.pulses):
            start = pulse.left
            stop = pulse.right
            channel = pulse.channel
            pmt_gain = self.config['gains'][channel]

            # Check the pulse properties have been computed
            if np.isnan(pulse.minimum):
                raise RuntimeError("Attempt to perform hitfinding on pulses whose properties have not been computed!")

            # Retrieve waveform as floats: needed to subtract baseline (which can be in-between ADC counts)
            w = pulse.raw_data.astype(np.float64)

            # Subtract the baseline and invert(so hits point up from baseline)
            w = self.reference_baseline - w
            w -= pulse.baseline

            # Don't do hitfinding in dead channels, pulse property computation was enough
            # Could refactor pulse property computation to separate plugin,
            # but that would mean waveform has to be converted to floats twice
            if pmt_gain == 0:
                continue

            # Check if the DAQ pulse was ADC-saturated (clipped)
            # This means the raw waveform dropped to 0,
            # i.e. we went digitizer_reference_baseline above the reference baseline
            # i.e. we went digitizer_reference_baseline - pulse.baseline above baseline
            # 0.5 is needed to avoid floating-point rounding errors to cause saturation not to be reported
            # Somehow happens only when you use simulated data -- apparently np.clip rounds slightly different
            is_saturated = pulse.maximum >= self.reference_baseline - pulse.baseline - 0.5

            # Compute thresholds based on noise level
            high_threshold = max(self.config['height_over_noise_high_threshold'] * pulse.noise_sigma,
                                 self.config['absolute_adc_counts_high_threshold'],
                                 - self.config['height_over_min_high_threshold'] * pulse.minimum)
            low_threshold = max(self.config['height_over_noise_low_threshold'] * pulse.noise_sigma,
                                self.config['absolute_adc_counts_low_threshold'],
                                - self.config['height_over_min_low_threshold'] * pulse.minimum)

            # Call the numba hit finder -- see its docstring for description
            n_hits_found = pulse.n_hits_found = find_intervals_above_threshold(w,
                                                                               float(high_threshold),
                                                                               float(low_threshold),
                                                                               hit_bounds_buffer,
                                                                               dynamic_low_threshold_coeff)

            # Only view the part of hit_bounds_buffer that contains hits found in this event
            # The rest of hit_bounds_buffer contains -1's or stuff from previous pulses
            hit_bounds_found = hit_bounds_buffer[:n_hits_found]

            # If no hits were found, this is a noise pulse: update the noise pulse count
            if n_hits_found == 0:
                event.noise_pulses_in[channel] += 1
                # Don't 'continue' to the next pulse! There's stuff left to do!
            elif n_hits_found >= self.max_hits_per_pulse:
                self.log.debug("Pulse %s-%s in channel %s has more than %s hits. "
                               "This usually indicates a zero-length encoding breakdown after a very large S2. "
                               "Further hits in this pulse have been ignored." % (start, stop, channel,
                                                                                  self.max_hits_per_pulse))

            # Store the found hits in the datastructure
            # Convert area, noise_sigma and height from adc counts -> pe
            adc_to_pe = dsputils.adc_to_pe(self.config, channel)
            noise_sigma_pe = pulse.noise_sigma * adc_to_pe

            build_hits(w, hit_bounds_found, hits_buffer,
                       adc_to_pe, channel, noise_sigma_pe, dt, start, pulse_i,
                       saturation_threshold=self.config['digitizer_reference_baseline'] - pulse.baseline - 0.5)
            hits = hits_buffer[:n_hits_found].copy()
            hits_per_pulse.append(hits)

            # Diagnostic plotting
            # TODO: Move to separate plugin, we have dict_group_by now

            # Do we need to show this pulse? If not: continue
            if self.make_diagnostic_plots == 'never':
                continue
            elif self.make_diagnostic_plots == 'tricky cases':
                # Always show pulse if noise level is very high
                if noise_sigma_pe < 0.5:
                    if len(hit_bounds_found) == 0:
                        # Show pulse if it nearly went over threshold
                        if not pulse.maximum > 0.8 * high_threshold:
                            continue
                    else:
                        # Show pulse if any of its hit nearly didn't go over threshold
                        if not any([event.all_hits[-(i+1)].height < 1.2 * high_threshold * adc_to_pe
                                   for i in range(len(hit_bounds_found))]):
                            continue
            elif self.make_diagnostic_plots == 'no hits':
                if len(hit_bounds_found) != 0:
                    continue
            elif self.make_diagnostic_plots == 'baseline shifts':
                if abs(pulse.baseline_increase) < 10:
                    continue
            elif self.make_diagnostic_plots == 'hits only':
                if len(hit_bounds_found) == 0:
                    continue
            elif self.make_diagnostic_plots == 'saturated':
                if not is_saturated:
                    continue
            else:
                if self.make_diagnostic_plots != 'always':
                    raise ValueError("Invalid make_diagnostic_plots option: %s!" % self.make_diagnostic_plots)

            plt.figure(figsize=(14, 10))
            data_for_title = (event.event_number, start, stop, channel)
            plt.title('Event %s, pulse %d-%d, Channel %d' % data_for_title)
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax1.set_position((.1, .1, .6, .85))
            ax2.set_position((.1, .1, .6, .85))
            ax1.set_xlabel("Sample number (%s ns)" % event.sample_duration)
            ax1.set_ylabel("ADC counts above baseline")
            ax2.set_ylabel("pe / sample")

            # Plot the signal and noise levels
            ax1.plot(w, drawstyle='steps-mid', label='Data')
            ax1.plot(np.ones_like(w) * high_threshold, '--', label='Threshold', color='red')
            ax1.plot(np.ones_like(w) * pulse.noise_sigma, ':', label='Noise level', color='gray')
            ax1.plot(np.ones_like(w) * pulse.minimum, '--', label='Minimum', color='orange')
            ax1.plot(np.ones_like(w) * low_threshold, '--', label='Boundary threshold', color='green')

            # Mark the hit ranges & center of gravity point
            for hit_i, hit in enumerate(hit_bounds_found):
                ax1.axvspan(hit[0] - 0.5, hit[1] + 0.5, color='red', alpha=0.2)

            # Make sure the y-scales match
            ax2.set_ylim(ax1.get_ylim()[0] * adc_to_pe, ax1.get_ylim()[1] * adc_to_pe)

            # Add pulse / hit information
            if len(hits) != 0:
                largest_hit = hits[np.argmax(hits['area'])]
                plt.figtext(0.75, 0.9, dedent("""
                            Pulse maximum: {pulse.maximum:.5g}
                            Pulse minimum: {pulse.minimum:.5g}
                              (both in ADCc above baseline)
                            Pulse baseline: {pulse.baseline}
                              (ADCc above reference baseline)
                            Baseline increase: {pulse.baseline_increase:.2f}

                            Gain in this PMT: {gain:.3g}

                            Largest hit info ({left}-{right}):
                            Area: {hit_area:.5g} pe
                            Height: {hit_height:.4g} pe
                            Saturated samples: {hit_n_saturated}
                            """.format(pulse=pulse,
                                       gain=self.config['gains'][pulse.channel],
                                       left=largest_hit['left']-pulse.left,
                                       right=largest_hit['right']-pulse.left,
                                       hit_area=largest_hit['area'],
                                       hit_height=largest_hit['height'],
                                       hit_n_saturated=largest_hit['n_saturated'])),
                            fontsize=14, verticalalignment='top')

            # Finish the plot, save, close
            leg = ax1.legend()
            leg.get_frame().set_alpha(0.5)
            plt.savefig(os.path.join(self.make_diagnostic_plots_in,
                                     'event%04d_pulse%05d-%05d_ch%03d.png' % data_for_title))
            plt.xlim(0, len(pulse.raw_data))
            plt.close()

        if len(hits_per_pulse):
            event.all_hits = np.concatenate(hits_per_pulse)
            self.log.debug("Found %d hits in %d pulses" % (len(event.all_hits), len(event.pulses)))
        else:
            self.log.warning("Event has no pulses??!")

        return event


@numba.jit(numba.void(numba.float64[:], numba.int64[:, :],
                      numba.from_dtype(datastructure.Hit.get_dtype())[:],
                      numba.float64, numba.int64, numba.float64, numba.int64, numba.int64, numba.int64, numba.float64),
           nopython=True)
def build_hits(w, hit_bounds,
               hits_buffer,
               adc_to_pe, channel, noise_sigma_pe, dt, start, pulse_i, saturation_threshold):
    """Populates hits_buffer with properties from hits indicated by hit_bounds.
        hit_bounds should be a numpy array of (left, right) bounds (inclusive) in w
    Returns nothing.
    """
    for hit_i in range(len(hit_bounds)):
        amplitude = -999.9
        argmax = -1
        area = 0.0
        center = 0.0
        deviation = 0.0
        saturation_count = 0
        left = hit_bounds[hit_i, 0]
        right = hit_bounds[hit_i, 1]
        for i, x in enumerate(w[left:right + 1]):
            if x > amplitude:
                amplitude = x
                argmax = i
            if x > saturation_threshold:
                saturation_count += 1
            area += x
            center += x * i
        center /= area
        for i, x in enumerate(w[left:right + 1]):
            deviation += x * abs(i - center)
        deviation /= area

        # Store the hit properties
        hits_buffer[hit_i].channel = channel
        hits_buffer[hit_i].found_in_pulse = pulse_i
        hits_buffer[hit_i].noise_sigma = noise_sigma_pe
        hits_buffer[hit_i].left = left + start
        hits_buffer[hit_i].right = right + start
        hits_buffer[hit_i].area = area * adc_to_pe
        hits_buffer[hit_i].sum_absolute_deviation = deviation
        hits_buffer[hit_i].center = (start + left + center) * dt
        hits_buffer[hit_i].height = w[argmax + left] * adc_to_pe
        hits_buffer[hit_i].index_of_maximum = start + left + argmax
        hits_buffer[hit_i].n_saturated = saturation_count
