import os
from textwrap import dedent

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from pax import plugin, datastructure, dsputils
from pax.recarray_tools import dict_group_by


class HitfinderDiagnosticPlots(plugin.TransformPlugin):
    """Make some useful diagnostic plots of the hitfinder's behaviour.

    The 'make_diagnostic_plots' option controls when to make these plots
    (since it takes a very long time to make them all).
        never - obvious. This is the default, which you always want to use this in production.
        always - obvious
        no hits - only pulses which have no hits
        hits only - only pulses which have hits
        tricky cases - pulses with hits that only just crossed the threshold, or no hits but almost one
        saturated - pulses whose ADC waveform is maximum (clips outside the dynamic range)

    'make_diagnostic_plots_in' sets the directory where the diagnostic plots are created.
    """

    def startup(self):
        c = self.config
        self.make_diagnostic_plots = c.get('make_diagnostic_plots', 'never')
        self.make_diagnostic_plots_in = c.get('make_diagnostic_plots_in', 'hitfinder_diagnostic_plots')
        self.reference_baseline = self.config['digitizer_reference_baseline']

        if self.make_diagnostic_plots != 'never':
            if not os.path.exists(self.make_diagnostic_plots_in):
                os.makedirs(self.make_diagnostic_plots_in)

    def transform_event(self, event):
        if self.make_diagnostic_plots == 'never':
            return event

        # Get the pulse-to-hit mapping
        # Note this relies on the hits being sorted by found_in_pulse
        # (probably shouldn't have rolled our own fake pandas for recarrays...)
        self.log.debug("Reconstructing hit/pulse mapping")
        hits_in_pulse = dict_group_by(event.all_hits, 'found_in_pulse')

        for pulse_i, pulse in enumerate(tqdm(event.pulses, desc='Making hitfinder diagnostic plots')):

            # Get the hits that were found in this pulse
            hits = hits_in_pulse.get(pulse_i, np.array([], datastructure.Hit.get_dtype()))

            # Reconstruct some variables we had in the hitfinder. Some code duplication unfortunately...
            # that's the price we pay for having diagnostic plotting cleanly separated from the hitfinder.
            start = pulse.left
            stop = pulse.right

            hit_bounds_found = np.vstack((hits['left'], hits['right'])).T
            hit_bounds_found -= start

            w = self.reference_baseline - pulse.raw_data.astype(np.float64) - pulse.baseline

            channel = pulse.channel
            adc_to_pe = dsputils.adc_to_pe(self.config, channel)
            noise_sigma_pe = pulse.noise_sigma * adc_to_pe
            threshold = pulse.hitfinder_threshold
            saturation_threshold = self.reference_baseline - pulse.baseline - 0.5
            is_saturated = pulse.maximum >= saturation_threshold

            # Do we need to show this pulse? If not: continue
            if self.make_diagnostic_plots == 'tricky cases':
                # Always show pulse if noise level is very high
                if noise_sigma_pe < 0.5:
                    if len(hit_bounds_found) == 0:
                        # Show pulse if it nearly went over threshold
                        if not pulse.maximum > 0.8 * threshold:
                            continue
                    else:
                        # Show pulse if any of its hit nearly didn't go over threshold
                        if not any([event.all_hits[-(i+1)].height < 1.2 * threshold * adc_to_pe
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
            elif self.make_diagnostic_plots == 'negative':
                # Select only pulses which had hits whose area and or height were originally negative
                # (the hitfinder helpfully capped them at 1e-9, so they're not actually negative...)
                if not (len(hits) and (np.any(hits['area'] < 1e-6) or np.any(hits['height'] < 1e-6))):
                    continue
            elif self.make_diagnostic_plots != 'always':
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
            ax1.plot(np.ones_like(w) * threshold, '--', label='Threshold', color='red')
            ax1.plot(np.ones_like(w) * pulse.noise_sigma, ':', label='Noise level', color='gray')
            ax1.plot(np.ones_like(w) * pulse.minimum, '--', label='Minimum', color='orange')

            # Mark the hit ranges & center of gravity point
            for hit_i, hit in enumerate(hit_bounds_found):
                ax1.axvspan(hit[0] - 0.5, hit[1] + 0.5, color='red', alpha=0.2)

            # Make sure the y-scales match
            ax2.set_ylim(ax1.get_ylim()[0] * adc_to_pe, ax1.get_ylim()[1] * adc_to_pe)

            # Add pulse / hit information
            if len(hits) != 0:
                largest_hit = hits[np.argmax(hits['area'])]
                plt.figtext(0.75, 0.98, dedent("""
                            Pulse maximum: {pulse.maximum:.5g}
                            Pulse minimum: {pulse.minimum:.5g}
                              (both in ADCc above baseline)
                            Pulse baseline: {pulse.baseline}
                              (in ADCc above reference baseline)
                            Baseline increase: {pulse.baseline_increase:.2f}

                            Gain in this PMT: {gain:.3g}

                            Noise level: {pulse.noise_sigma:.2f} ADCc
                            Hitfinder threshold: {pulse.hitfinder_threshold} ADCc

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

        return event
