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
                adc_to_pe = dsputils.adc_to_pe(self.config, pulse.channel)
                gain = self.config['gains'][channel]

                if detector == 'tpc':
                    if channel in self.config['channels_top']:
                        sum_w = event.get_sum_waveform('tpc_top')
                    else:
                        sum_w = event.get_sum_waveform('tpc_bottom')
                else:
                    sum_w = event.get_sum_waveform(detector)

            # Don't consider dead channels
            if gain == 0:
                continue

            baseline_to_subtract = self.config['digitizer_reference_baseline'] - pulse.baseline

            w = baseline_to_subtract - pulse.raw_data.astype(np.float32)
            w *= adc_to_pe

            sum_w_raw = event.get_sum_waveform(detector+'_raw').samples
            sum_w_raw[pulse.left:pulse.right+1] += w

            hits = hits_per_pulse.get(pulse_i, None)
            if hits is None:
                continue

            w_hits_only = w.copy()
            zero_waveform_outside_hits(w_hits_only,
                                       hits['left'] - pulse.left,
                                       hits['right'] - pulse.left)
            sum_w.samples[pulse.left:pulse.right+1] += w_hits_only

        # Sum the tpc top and bottom tpc waveforms
        event.get_sum_waveform('tpc').samples = event.get_sum_waveform('tpc_top').samples + \
            event.get_sum_waveform('tpc_bottom').samples

        return event


@numba.jit(numba.void(numba.float32[:], numba.int64[:], numba.int64[:]), nopython=True, cache=True)
def zero_waveform_outside_hits(w, left, right,):
    """Assumes hits don't overlap, and are sorted from left to right"""
    if len(left) == 0:
        return
    w[:left[0]] = 0
    for i in range(1, len(left)):
        w[right[i-1]+1:left[i]] = 0
    w[right[-1]+1:] = 0
