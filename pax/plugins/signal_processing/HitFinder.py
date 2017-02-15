import numpy as np
import numba

import os

from pax import plugin, datastructure, dsputils
from pax.dsputils import find_intervals_above_threshold, extend_intervals, build_hits, find_split_points


class FindHits(plugin.TransformPlugin):
    """Finds hits in pulses, proceeding in a few stages
        1. The waveform is baseline-corrected (with baseline as determined by PulseProperties.PulseProperties)
        2. Hits are found based on a threshold, which can be set on three different quantities (see below)
        3. Hits are extended left and right depending on the left_extension and right_extension settings.

    Types of thresholds:

    1) Height over noise threshold
       This threshold operates on the height above baseline / noise level
       The noise level is deteremined in each pulse as
         (<(w - baseline)**2>)**0.5
       with the average running *only* over samples < baseline!

    2) Absolute ADC counts above baseline
       If there is measurable noise in the waveform, you don't want the thresholds to fall to 0 and crash pax.
       This happens for pulses constant on initial_baseline_samples.
       Please use this as a deep fallback only, unless you know what mess you're getting yourself into!

    3) - Height / minimum
       Using this threshold protects you against sudden up & down fluctuations, especially in large pulses
       However, keep in mind that one large downfluctuation will cause a sensitivity loss throughout an entire pulse;
       if you have ZLE off, this may be a bad idea...
       This threshold operates on the (-) height above baseline / height below baseline of the lowest sample in pulse

    The threshold is computed and stored per pulse.
    Here high/positive means 'signal like' (i.e. the PMT voltage becomes more negative). If multiple thresholds are
    given, the highest is used.

    Edge case:

    *) After large peaks the zero-length encoding can fail, making a huge pulse with many hits.
       If more than max_hits_per_pulse hits are found in a pulse, the rest will be ignored.
       If this threshold is set too low, you risk missing some hits in such events.
       If set too high, it will degrade performance. Don't set to infinity, we need to allocate memory for this...

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

        self.always_find_single_hit = self.config.get('always_find_single_hit')

    def transform_event(self, event):
        dt = self.config['sample_duration']
        hits_per_pulse = []

        left_extension = self.config['left_extension'] // dt
        right_extension = self.config['right_extension'] // dt

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

            # Compute hitfinder threshold to use
            # Rounding down is ok, since hitfinder uses >, not >= for threshold crossing testing.
            pulse.hitfinder_threshold = int(max(self.config['height_over_noise_threshold'] * pulse.noise_sigma,
                                                self.config['absolute_adc_counts_threshold'],
                                                - self.config['height_over_min_threshold'] * pulse.minimum))
            if self.always_find_single_hit:
                # The config specifies a single range to integrate. Useful for gain calibration
                hit_bounds_buffer[0] = self.always_find_single_hit
                n_hits_found = 1
            else:
                # Call the numba hit finder -- see its docstring for description
                n_hits_found = find_intervals_above_threshold(w,
                                                              threshold=float(pulse.hitfinder_threshold),
                                                              result_buffer=hit_bounds_buffer)

            # Only view the part of hit_bounds_buffer that contains hits found in this event
            # The rest of hit_bounds_buffer contains -1's or stuff from previous pulses
            pulse.n_hits_found = n_hits_found
            hit_bounds_found = hit_bounds_buffer[:n_hits_found]

            if self.always_find_single_hit:
                central_bounds = hit_bounds_found
            else:
                # Split the hits at sufficiently salient local minima
                split_points = []
                for hit_i in range(n_hits_found):
                    l = hit_bounds_found[hit_i,0]
                    r = hit_bounds_found[hit_i,1]
                    split_points += [x + l for x in find_split_points(w[l:r], 0.5, 1.5)]

                if len(split_points):
                    split_points = np.array(split_points)

                    hit_bounds_found = np.sort(np.concatenate([hit_bounds_found.ravel(),
                                                               split_points,
                                                               split_points + 1])).reshape(-1, 2).astype(np.int)

                    pulse.n_hits_found = n_hits_found = len(hit_bounds_found)

                # Extend the boundaries of each hit, to be sure we integrate everything.
                # The original bounds are preserved: they are used in clustering
                central_bounds = hit_bounds_found.copy()
                extend_intervals(w, hit_bounds_found, left_extension, right_extension)

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

            # If the DAQ pulse was ADC-saturated (clipped), the raw waveform dropped to 0,
            # i.e. we went digitizer_reference_baseline above the reference baseline
            # i.e. we went digitizer_reference_baseline - pulse.baseline above baseline
            # 0.5 is needed to avoid floating-point rounding errors to cause saturation not to be reported
            # Somehow happens only when you use simulated data -- apparently np.clip rounds slightly different
            saturation_threshold = self.reference_baseline - pulse.baseline - 0.5

            build_hits(w, hit_bounds_found, hits_buffer,
                       adc_to_pe, channel, noise_sigma_pe, dt, start, pulse_i, saturation_threshold, central_bounds)
            hits = hits_buffer[:n_hits_found].copy()
            hits_per_pulse.append(hits)

        if len(hits_per_pulse):
            event.all_hits = np.concatenate(hits_per_pulse)
            self.log.debug("Found %d hits in %d pulses" % (len(event.all_hits), len(event.pulses)))
        else:
            self.log.warning("Event has no pulses??!")

        return event
