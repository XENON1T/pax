import numpy as np
from pax import plugin, units, datastructure


class JoinAndConvertWaveforms(plugin.TransformPlugin):

    def startup(self):
        c = self.config
        self.gains = c['gains']
        # Conversion factor from converting from ADC counts -> pmt-electrons/bin
        # Still has to be divided by PMT gain to get pe/bin
        self.conversion_factor = c['digitizer_t_resolution'] * c['digitizer_voltage_range'] / (
            2 ** (c['digitizer_bits']) * c['pmt_circuit_load_resistor']
            * c['external_amplification'] * units.electron_charge
        )

    def transform_event(self, event):

        pmts = 1+max(self.config['pmts_veto'])   # TODO: really??
        event.pmt_waveforms = np.zeros((pmts, event.length()))
        # Should these go into event class?
        baselines = np.zeros(pmts)
        baseline_stdevs = np.zeros(pmts)

        for channel, waveform_occurrences in event.occurrences.items():

            # Deal with unknown and dead channels
            if channel not in self.gains:
                self.log.error('Gain for channel %s is not specified!' % channel)
                continue
            if self.gains[channel] == 0:
                self.log.debug('Gain for channel %s is 0, assuming it is dead.' % channel)
                continue

            # Compute the baseline
            # Get all samples we can use for baseline computation
            baseline_samples = []
            for i, (start_index, pulse_wave) in enumerate(waveform_occurrences):
                # Check for pulses starting right after previous ones: don't use these for baseline computation
                if i != 0 and start_index == waveform_occurrences[i - 1][0] + len(waveform_occurrences[i - 1][1]):
                    continue
                # Pulses that start at the very beginning are also off-limits, they may not have
                if start_index == 0:
                    continue
                baseline_samples.extend(pulse_wave[:self.config['baseline_samples_at_start_of_pulse']])
            # Use the mean of the median 20% for the baseline: ensures outliers don't skew computed baseline
            baselines[channel] = np.mean(sorted(baseline_samples)[
                int(0.4 * len(baseline_samples)):int(0.6 * len(baseline_samples))
            ])
            baseline_stdevs[channel] = np.std(baseline_samples)

            # Convert and store the PMT waveform in the right place
            for i, (start_index, pulse_wave) in enumerate(waveform_occurrences):
                event.pmt_waveforms[channel][
                    start_index:start_index + len(pulse_wave)
                ] = (baselines[channel] - pulse_wave) * self.conversion_factor / self.gains[channel]

        return event


class SumWaveforms(plugin.TransformPlugin):

    def startup(self):
        self.channel_groups = {'top': self.config['pmts_top'],
                               'bottom': self.config['pmts_bottom'],
                               'veto': self.config['pmts_veto']}

    def transform_event(self, event):
        # Compute summed waveforms
        for group, members in self.channel_groups.items():
            event.waveforms.append(datastructure.Waveform({
                'samples':  np.sum(event.pmt_waveforms[[list(members)]], axis=0),
                'name':     group,
                'pmt_list': members
            }))
        event.waveforms.append(datastructure.Waveform({
            'samples':  event.get_waveform('top').samples + event.get_waveform('bottom').samples,
            'name':     'tpc',
            'pmt_list': (self.channel_groups['top'] | self.channel_groups['bottom'])
        }))
        return event


class S2Filter(plugin.TransformPlugin):

    def transform_event(self, event):
        input_w = event.get_waveform('tpc')
        event.waveforms.append(datastructure.Waveform({
            'name':     'filtered_for_s2',
            'samples':  np.convolve(input_w.samples, self.config['normalized_filter_ir'], 'same'),
            'pmt_list': input_w.pmt_list,
        }))

        return event


class FindPeaks(plugin.TransformPlugin):
    def transform_event(self, event):
        filtered = event.get_waveform('filtered_for_s2').samples
        unfiltered = event.get_waveform('tpc').samples
        candidates = self.intervals_above_threshold(filtered, self.config['s2_threshold'])
        for itv_left, itv_right in candidates:
            self.log.debug("S2 candidate interval %s-%s" % (itv_left, itv_right))
            max_idx = itv_left + np.argmax(unfiltered[itv_left:itv_right + 1])
            #TODO: better extent determination
            left = itv_left
            right = itv_right
            self.log.debug("S2 candidate peak %s-%s-%s" % (left, max_idx, right))
            event.S2s.append(datastructure.Peak({
                'area':             np.sum(unfiltered[left:right]),
                'index_of_maximum': max_idx,
                'height':           unfiltered[max_idx],
                'left':             left,
                'right':            right,
            }))
        return event

    @staticmethod
    def intervals_above_threshold(signal, threshold):
        """Finds all intervals in signal above threshold"""
        above0 = np.clip(np.sign(signal - threshold), 0, float('inf'))
        above0[-1] = 0      # Last sample is always an end. Also prevents edge cases due to rolling it over.
        above0_next = np.roll(above0, 1)
        cross_above = np.sort(np.where(above0 - above0_next == 1)[0])
        cross_below = np.sort(np.where(above0 - above0_next == -1)[0] - 1)
        # Assuming each interval's left <= right, we can simply split sorted(lefts+rights) in pairs:
        return list(zip(*[iter(sorted(list(cross_above) +list(cross_below)))] * 2))