import numpy as np
import numba

from pax import plugin, datastructure, recarray_tools, dsputils


class SumWaveform(plugin.TransformPlugin):

    def startup(self):
        self.detector_by_channel = dsputils.get_detector_by_channel(self.config)

    def transform_event(self, event):

        # Initialize empty waveforms for each detector
        # One with only hits, one with raw data
        for postfix in ('', '_raw'):
            for detector, chs in self.config['channels_in_detector'].items():
                event.sum_waveforms.append(datastructure.SumWaveform(
                    samples=np.zeros(event.length(), dtype=np.float32),
                    name=detector + postfix,
                    channel_list=np.array(list(chs), dtype=np.uint16),
                    detector=detector
                ))

        # Make dictionary mapping pulse -> non-rejected hits found in pulse
        # Assumes hits are still sorted by pulse
        event.all_hits = np.sort(event.all_hits, order='found_in_pulse')
        hits_per_pulse = recarray_tools.dict_group_by(event.all_hits[True ^ event.all_hits['is_rejected']],
                                                      'found_in_pulse')

        # Add top and bottom tpc sum waveforms
        for q in ('top', 'bottom'):
            event.sum_waveforms.append(datastructure.SumWaveform(
                samples=np.zeros(event.length(), dtype=np.float32),
                name='tpc_%s' % q,
                channel_list=np.array(self.config['channels_%s' % q], dtype=np.uint16),
                detector='tpc'
            ))

        for pulse_i, pulse in enumerate(event.pulses):
            channel = pulse.channel

            # Do some initialization only when we switch channel
            # The 'current_channel' variable can't be pulled outside the loop, that would hurt performance
            # (trust me, try it and time it)
            if pulse_i == 0 or channel != current_channel:      # noqa
                current_channel = channel                       # noqa
                detector = self.detector_by_channel[channel]
                adc_to_pe = dsputils.adc_to_pe(self.config, channel)

                if detector == 'tpc':
                    if channel in self.config['channels_top']:
                        sum_w = event.get_sum_waveform('tpc_top')
                    else:
                        sum_w = event.get_sum_waveform('tpc_bottom')
                else:
                    sum_w = event.get_sum_waveform(detector)

            # Don't consider dead channels
            if self.config['gains'][channel] == 0:
                continue

            # Get the pulse waveform in pe/bin
            baseline_to_subtract = self.config['digitizer_reference_baseline'] - pulse.baseline
            w = baseline_to_subtract - pulse.raw_data.astype(np.float64)
            w *= adc_to_pe

            sum_w_raw = event.get_sum_waveform(detector+'_raw').samples
            sum_w_raw[pulse.left:pulse.right+1] += w

            hits = hits_per_pulse.get(pulse_i, None)
            if hits is None:
                continue

            w_hits_only = w.copy()
            # Obtain an array of same length as w, indicating whether the sample is in a hit or not
            mask = np.zeros(len(w), dtype=np.bool)
            set_if_in_ranges(mask, hits['left'] - pulse.left, hits['right'] - pulse.left)
            w_hits_only[True ^ mask] = 0

            sum_w.samples[pulse.left:pulse.right+1] += w_hits_only

        # Sum the tpc top and bottom tpc waveforms
        event.get_sum_waveform('tpc').samples = event.get_sum_waveform('tpc_top').samples + \
            event.get_sum_waveform('tpc_bottom').samples

        return event


@numba.jit(numba.void(numba.bool_[:], numba.int64[:], numba.int64[:]), nopython=True)
def set_if_in_ranges(w, left, right):
    """Set samples in w to True between ranges indicated by left and right (both inclusive)"""
    for i in range(len(left)):
        w[left[i]:right[i]+1] = True
