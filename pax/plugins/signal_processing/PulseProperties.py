import numba
import numpy as np

from pax import plugin


class PulseProperties(plugin.TransformPlugin):
    """Compute pulse properties such as the baseline and noise level.
    Optionally, deletes the left and right extreme ends of the pulses in large events to reduce datasets.
    Note the raw data of the pulse is not in any way modified: the computed baseline is not actually subtracted.

    If the raw data already has the pulse properties pre-computed, no action is taken.
    """
    warning_given = False

    def transform_event(self, event):
        # Local variables are marginally faster to access in inner loop, so we don't put these in startup.
        reference_baseline = self.config['digitizer_reference_baseline']

        n_baseline = self.config.get('baseline_samples', 50)

        shrink_data_threshold = self.config.get('shrink_data_threshold', float('inf'))
        shrink_data_samples = self.config.get('shrink_data_samples', n_baseline)

        n_pulses = len(event.pulses)
        warning_given = self.warning_given

        for pulse_i, pulse in enumerate(event.pulses):
            if not np.isnan(pulse.minimum):
                if not warning_given:
                    self.log.info("Pulse properties have already been computed, doing nothing.")
                    self.warning_given = True
                return event

            # Retrieve waveform as floats: needed to subtract baseline (which can be in between ADC counts)
            w = pulse.raw_data.astype(np.float64)

            # Subtract reference baseline, invert (so hits point up from baseline)
            # This is convenient so we don't have to reinterpret min, max, etc
            w = reference_baseline - w

            _results = compute_pulse_properties(w, n_baseline)
            pulse.baseline, pulse.baseline_increase, pulse.noise_sigma, pulse.minimum, pulse.maximum = _results

            if n_pulses > shrink_data_threshold:
                # Remove the start and end of each pulse, which don't contain much useful information, but
                # takes up a lot of space
                pulse.raw_data = pulse.raw_data[shrink_data_samples:-shrink_data_samples]
                pulse.right -= n_baseline
                pulse.left += n_baseline

                # Store some "advanced" pulse properties as ints rather than floats to save space
                # TODO: disabled, seems to freak out ROOT output??
                # Do NOT round noise sigma as an int, it could come out to 0, and precision matters here
                # pulse.maximum = int(round(pulse.maximum))
                # pulse.minimum = int(round(pulse.minimum))
                # pulse.baseline = int(round(pulse.baseline))
                # pulse.baseline_increase = int(round(pulse.baseline_increase))

        return event


@numba.jit(numba.typeof((1.0, 1.0, 1.0, 1.0, 1.0))(numba.float64[:], numba.int64),
           nopython=True)
def compute_pulse_properties(w, baseline_samples):
    """Compute basic pulse properties quickly
    :param w: Raw pulse waveform in ADC counts
    :param baseline_samples: number of samples to use for baseline computation at the start of the pulse
    :return: (baseline, baseline_increase, noise_sigma, min, max);
      baseline is the average of the first baseline_samples in the pulse
      baseline_increase = baseline_after - baseline_before
      min and max relative to baseline
      noise_sigma is the std of samples below baseline
    Does not modify w. Does not assume anything about inversion of w!!
    """
    # Compute the baseline before and after the self-trigger
    baseline = 0.0
    baseline_samples = min(baseline_samples, len(w))
    for x in w[:baseline_samples]:
        baseline += x
    baseline /= baseline_samples

    baseline_after = 0.0
    for x in w[-baseline_samples:]:
        baseline_after += x
    baseline_after /= baseline_samples

    baseline_increase = baseline_after - baseline

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

    return baseline, baseline_increase, noise, min_a - baseline, max_a - baseline
