import numpy as np
from pax import plugin
from pax.dsputils import adc_to_pe


class DesaturatePulses(plugin.TransformPlugin):

    def startup(self):
        self.reference_baseline = self.config['digitizer_reference_baseline']

    def transform_event(self, event):
        tpc_channels = np.array(self.config['channels_in_detector']['tpc'])
        is_saturated = np.array([self.is_saturated(p) for p in event.pulses])
        reference_region_samples = self.config.get('reference_region_samples', 10)

        for pulse_i, pulse in enumerate(event.pulses):
            if not is_saturated[pulse_i] or pulse.channel not in tpc_channels:
                continue

            # Find all pulses in TPC channels that overlap in time, but didn't saturate
            other_pulses = [p for i, p in enumerate(event.pulses)
                            if p.left < pulse.right and p.right > p.left and not is_saturated[i] and
                            p.channel in tpc_channels]

            if not len(other_pulses):
                # Rare case where no other pulses available, one channel going crazy?
                continue

            # Compute the (gain-weighted) sum waveform of the non-saturated pulses
            min_left = min([p.left for p in other_pulses + [pulse]])
            max_right = max([p.right for p in other_pulses + [pulse]])
            sumw = np.zeros(max_right - min_left + 1)
            for p in other_pulses:
                offset = p.left - min_left
                sumw[offset:offset + len(p.raw_data)] += self.waveform_in_pe(p)

            # Crop it to include just the part that overlaps with this pulse
            offset = pulse.left - min_left
            sumw = sumw[offset:offset + len(pulse.raw_data)]

            # Where is the current pulse saturated?
            saturated = pulse.raw_data <= 0            # Boolean array, True if sample is saturated

            # Get a region of nonsaturated samples, a few samples to the left and right of the saturated region(s)
            reference_region = True ^ saturated
            reference_region &= np.arange(len(saturated)) > (np.where(saturated)[0].min() - reference_region_samples)
            reference_region &= np.arange(len(saturated)) < (np.where(saturated)[0].max() + reference_region_samples)

            # Compute the ratio of this channel's waveform / the nonsaturated waveform in the reference region
            w = self.waveform_in_pe(pulse)
            ratio = w[reference_region].sum()/sumw[reference_region].sum()

            # Reconstruct the waveform in the saturated region according to this ratio.
            # The waveform should never be reduced due to this (then we are sure the correction is making things worse)
            w[saturated] = np.clip(sumw[saturated] * ratio, w[saturated], float('inf'))

            # Convert back to raw ADC counts and store the corrected waveform
            # Note this changes the type of pulse.w from int16 to float64: we don't have a choice,
            # int16 probably can't contain the large amplitudes we may be putting in.
            # As long as the raw data isn't saved again after applying this correction, this should be no problem
            # (as in later code converting to floats is anyway the first step).
            w /= adc_to_pe(self.config, pulse.channel)
            w = self.reference_baseline - w - pulse.baseline

            pulse.raw_data = w

        return event

    def waveform_in_pe(self, p):
        """Return waveform in pe/bin above baseline of a pulse"""
        w = self.reference_baseline - p.raw_data.astype(np.float) - p.baseline
        w *= adc_to_pe(self.config, p.channel)
        return w

    def is_saturated(self, p):
        """Return if a pulse is saturated"""
        return p.maximum >= self.reference_baseline - p.baseline - 0.5
