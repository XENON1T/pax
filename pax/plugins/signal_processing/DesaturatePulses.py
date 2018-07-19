import numpy as np
from pax import plugin
from pax.dsputils import adc_to_pe


class DesaturatePulses(plugin.TransformPlugin):
    """Estimates the waveform shape in channels that go beyond the digitizer's dynamic range, using
    the other channels' waveform shape as a template.
    pulse.w will be changed from int16 to float64

    See Fei & Yuehan's note: media=xenon:feigao:xenon1t_background_comparison_jan2017.html
    """

    def startup(self):
        self.reference_baseline = self.config['digitizer_reference_baseline']

    def transform_event(self, event):
        tpc_channels = np.array(self.config['channels_in_detector']['tpc'])

        # Boolean array, tells us which pulses are saturated
        is_saturated = np.array([p.maximum >= self.reference_baseline - p.baseline - 0.5
                                 for p in event.pulses])

        for pulse_i, pulse in enumerate(event.pulses):
            # Consider only saturated pulses in the TPC
            if not is_saturated[pulse_i] or pulse.channel not in tpc_channels:
                continue

            # Where is the current pulse saturated?
            saturated = pulse.raw_data <= 0            # Boolean array, True if sample is saturated
            _where_saturated_all = np.where(saturated)[0]

            # Split saturation if there is long enough non-saturated samples in between
            _where_saturated_diff = np.diff(_where_saturated_all, n=1)
            _where_saturated_diff = np.where(_where_saturated_diff > self.config['reference_region_samples'])[0]
            _where_saturated_list = np.split(_where_saturated_all, _where_saturated_diff+1)

            # Find all pulses in TPC channels that overlap with the saturated & reference region
            other_pulses = [p for i, p in enumerate(event.pulses)
                            if p.left < pulse.right and p.right > pulse.left and
                            not is_saturated[i] and
                            p.channel in tpc_channels and
                            p.channel not in self.config['large_after_pulsing_channels']]

            if not len(other_pulses):
                # Rare case where no other pulses available, one channel going crazy?
                continue

            for peak_i, _where_saturated in enumerate(_where_saturated_list):
                try:
                    first_saturated = _where_saturated.min()
                    last_saturated = _where_saturated.max()
                except (ValueError, RuntimeError, TypeError, NameError):
                    continue

                # Select a reference region just before the start of the saturated region
                reference_slice = slice(max(0, first_saturated - self.config['reference_region_samples']),
                                        first_saturated)

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

                # Compute the ratio of this channel's waveform / the nonsaturated waveform in the reference region
                w = self.waveform_in_pe(pulse)
                if len(sumw[reference_slice][sumw[reference_slice] > 1]) \
                        < self.config['reference_region_samples_treshold']:
                    # the pulse is saturated, but there are not enough reference samples to get a good ratio
                    # This actually distinguished between S1 and S2 and will only correct S2 signals
                    continue

                ratio = w[reference_slice].sum()/sumw[reference_slice].sum()

                # not < is preferred over >, since it will catch nan
                if not ratio < self.config.get('min_reference_area_ratio', 1):
                    # The pulse is saturated, but insufficient information is available in the other channels
                    # to reliably reconstruct it
                    continue

                if len(w[reference_slice][w[reference_slice] > 1]) < self.config['reference_region_samples_treshold']:
                    # the pulse is saturated, but there are not enough reference samples to get a good ratio
                    # This actually distinguished between S1 and S2 and will only correct S2 signals
                    continue

                # Finding individual section of wf for each peak
                # First end before the reference region of next peak
                if peak_i+1 == len(_where_saturated_list):
                    end = len(w)
                else:
                    end = _where_saturated_list[peak_i+1][0]-self.config['reference_region_samples']

                # Second end before the first upwards turning point
                v = sumw[last_saturated: end]
                conv = np.ones(self.config['convolution_length'])/self.config['convolution_length']
                v = np.convolve(conv, v, mode='same')
                dv = np.diff(v, n=1)
                # Choose +2 pe/ns instead 0 to avoid ending on the flat waveform
                turning_point = np.where((np.hstack((dv, -10)) > 2) & (np.hstack((10, dv)) <= 2))[0]

                if len(turning_point) > 0:
                    end = last_saturated + turning_point[0]

                # Reconstruct the waveform in the saturated region according to this ratio.
                # The waveform should never be reduced due to this (then the correction is making things worse)
                saturated_to_correct = np.arange(int(first_saturated), int(end))
                w[saturated_to_correct] = np.clip(sumw[saturated_to_correct] * ratio, 0, float('inf'))

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
