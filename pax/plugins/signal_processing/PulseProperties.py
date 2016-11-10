import numba
import numpy as np

from pax import plugin


class PulseProperties(plugin.TransformPlugin):
    """Compute pulse properties such as the baseline and noise level.
    Optionally, deletes the part of the raw data used for computing these from each pulse for large events
    (to reduce the data size).
    Note the raw data of the pulse is not otherwise modified (i.e. no baseline substraction is applied).

    If the raw data already has the pulse properties pre-computed, no action is taken.
    """
    warning_given = False

    def transform_event(self, event):
        # Local variables are marginally faster to access in inner loop, so we don't put these in startup.
        reference_baseline = self.config['digitizer_reference_baseline']
        n_baseline = self.config.get('baseline_samples', 50)
        shrink_data_threshold = self.config.get('shrink_data_threshold', float('inf'))
        n_pulses = len(event.pulses)
        warning_given = self.warning_given

        for pulse_i, pulse in enumerate(event.pulses):
            if not np.isnan(pulse.minimum):
                if not warning_given:
                    self.log.warning("Pulse properties have been pre-computed, doing nothing.")
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
                # Remove the pieces of raw data used for baselining
                pulse.raw_data = pulse.raw_data[n_baseline:-n_baseline]
                pulse.left += n_baseline
                pulse.right -= n_baseline

                # Store the six "advanced" pulse properties as ints rather than floats to save space
                pulse.maximum = int(pulse.maximum)
                pulse.minimum = int(pulse.minimum)
                pulse.noise_sigma = int(pulse.noise_sigma)
                pulse.baseline = int(pulse.baseline)
                pulse.baseline_increase = int(pulse.baseline_increase)

        return event


@numba.jit(numba.typeof((1.0, 1.0, 1.0, 1.0, 1.0))(numba.float64[:], numba.int64),
           nopython=True)
def compute_pulse_properties(w, baseline_samples):
    """Compute basic pulse properties quickly
    :param w: Raw pulse waveform in ADC counts
    :param baseline_samples: number of samples to use for baseline computation at start and end of pulse
    :return: (baseline, baseline_increase, noise_sigma, min, max);
      baseline is the largest baseline of the bl computed at the start and the bl computed at the end
      baseline_increase = baseline_after - baseline_before
      min and max relative to baseline
      noise_sigma is the std of samples below baseline
    Does not modify w. Does not assume anything about inversion of w!!
    """
    # Compute the baseline before and after the self-trigger
    baseline_before = 0.0
    baseline_samples = min(baseline_samples, len(w))
    for x in w[:baseline_samples]:
        baseline_before += x
    baseline_before /= baseline_samples

    baseline_after = 0.0
    for x in w[-baseline_samples:]:
        baseline_after += x
    baseline_after /= baseline_samples

    baseline = max(baseline_before, baseline_after)
    baseline_increase = baseline_after - baseline_before

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
