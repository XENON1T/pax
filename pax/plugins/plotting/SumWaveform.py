import numpy as np

from pax import plugin, datastructure, units, utils


class SumWaveform(plugin.TransformPlugin):

    def startup(self):
        # Build the channel -> detector lookup dict
        # Only used for summing waveforms
        self.detector_by_channel = {}
        for name, chs in self.config['channels_in_detector'].items():
            for ch in chs:
                self.detector_by_channel[ch] = name

    def transform_event(self, event):
        # Initialize empty waveforms for each detector
        # One with only hits, one with raw data
        for postfix in ('', '_raw'):
            for detector, chs in self.config['channels_in_detector'].items():
                event.sum_waveforms.append(datastructure.SumWaveform(
                    samples=np.zeros(event.length()),
                    name=detector + postfix,
                    channel_list=np.array(list(chs), dtype=np.uint16),
                    detector=detector
                ))

        # Build the raw sum waveform
        for pulse in event.pulses:
            channel = pulse.channel
            detector = self.detector_by_channel[channel]

            # Don't consider dead channels
            if self.config['gains'][channel] == 0:
                continue

            if self.config['subtract_reference_baseline_only_for_raw_waveform']:
                baseline_to_subtract = self.config['digitizer_reference_baseline']
            else:
                baseline_to_subtract = self.config['digitizer_reference_baseline'] + pulse.baseline

            w = baseline_to_subtract - pulse.raw_data.astype(np.float64)

            adc_to_pe = utils.adc_to_pe(self.config, pulse.channel)

            sum_w_raw = event.get_sum_waveform(detector+'_raw').samples
            sum_w_raw[pulse.left:pulse.right+1] += w * adc_to_pe

        # Build the hits-only sum waveform
        for hit in event.all_hits:
            channel = hit.channel
            detector = self.detector_by_channel[channel]

            # Don't include rejected hits - this is known only after clustering
            if hit.is_rejected:
                continue

            pulse = event.pulses[hit.found_in_pulse]

            # Retrieve the waveform, subtract baseline, invert
            left_in_pulse = hit.left - pulse.left
            right_in_pulse = hit.right - pulse.left
            w = (self.config['digitizer_reference_baseline'] + pulse.baseline) - \
                pulse.raw_data[left_in_pulse:right_in_pulse+1].astype(np.float64)

            adc_to_pe = utils.adc_to_pe(self.config, hit.channel)

            sum_w = event.get_sum_waveform(detector).samples
            sum_w[hit.left:hit.right+1] += w * adc_to_pe

        return event
