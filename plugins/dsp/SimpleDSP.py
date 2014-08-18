import numpy as np
from pax import plugin, units, datastructure, dsputils


class BuildWaveforms(plugin.TransformPlugin):

    def startup(self):
        c = self.config
        self.gains = c['gains']
        # Conversion factor from converting from ADC counts -> pmt-electrons/bin
        # Still has to be divided by PMT gain to get pe/bin
        self.conversion_factor = c['digitizer_t_resolution'] * c['digitizer_voltage_range'] / (
            2 ** (c['digitizer_bits']) * c['pmt_circuit_load_resistor']
            * c['external_amplification'] * units.electron_charge
        )
        self.channel_groups = {'top': self.config['pmts_top'],
                               'bottom': self.config['pmts_bottom'],
                               'veto': self.config['pmts_veto']}

    def transform_event(self, event):
        pmts = 1 + max(self.config['pmts_veto'])   # TODO: really??
        # Initialize empty waveforms
        event.pmt_waveforms = np.zeros((pmts, event.length()))
        for group, members in self.channel_groups.items():
            event.waveforms.append(datastructure.Waveform({
                'samples':  np.zeros(event.length()),
                'name':     group,
                'pmt_list': self.crazy_type_conversion(members),
            }))
        # Should these go into event class?
        object.__setattr__(event, 'baselines', np.zeros(pmts))
        object.__setattr__(event, 'baseline_stdevs', np.zeros(pmts))

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
            event.baselines[channel] = np.mean(sorted(baseline_samples)[
                int(0.4 * len(baseline_samples)):int(0.6 * len(baseline_samples))
            ])
            event.baseline_stdevs[channel] = np.std(baseline_samples)

            # Convert and store the PMT waveform in the right place
            for i, (start_index, pulse_wave) in enumerate(waveform_occurrences):
                end_index = start_index + len(pulse_wave)   # Well, not really index... index-1
                corrected_pulse = (event.baselines[channel] - pulse_wave) * self.conversion_factor / self.gains[channel]
                event.pmt_waveforms[channel][start_index:end_index] = corrected_pulse
                # Add occurrence to all appropriate summed waveforms
                for group, members in self.channel_groups.items():
                    if channel in members:
                        event.get_waveform(group).samples[start_index:end_index] += corrected_pulse
        # Add the tpc waveform: sum of top and bottom
        event.waveforms.append(datastructure.Waveform({
            'samples':  event.get_waveform('top').samples + event.get_waveform('bottom').samples,
            'name':     'tpc',
            'pmt_list': self.crazy_type_conversion(self.channel_groups['top'] | self.channel_groups['bottom'])
        }))
        return event

    @staticmethod
    def crazy_type_conversion(x):
        return np.array(list(x), dtype=np.uint16)


class S2Filter(plugin.TransformPlugin):

    def startup(self):
        filter_sigma = self.config['gauss_filter_fwhm']/2.35482
        x = np.linspace(-2,2,num=(1+4*filter_sigma/self.config['digitizer_t_resolution']))
        self.filter_ir = 1/(filter_sigma*np.sqrt(2*np.pi)) * np.exp(-x**2 / (2*filter_sigma**2))
        self.filter_ir /= np.sum(self.filter_ir)

    def transform_event(self, event):
        input_w = event.get_waveform('tpc')
        event.waveforms.append(datastructure.Waveform({
            'name':     'filtered_for_s2',
            'samples':  np.array(
                np.convolve(input_w.samples, self.filter_ir, 'same')
            ),
            'pmt_list': input_w.pmt_list,
        }))

        return event


class FindS2s(plugin.TransformPlugin):

    def transform_event(self, event):
        filtered = event.get_waveform('filtered_for_s2').samples
        unfiltered = event.get_waveform('tpc').samples
        # Find intervals above threshold in filtered waveform
        candidate_intervals = dsputils.intervals_above_threshold(filtered, self.config['s2_threshold'])
        # Find peaks in intervals
        peaks = dsputils.find_peaks_in_intervals(unfiltered, candidate_intervals, 's2')
        # Compute peak extents - on FILTERED waveform!
        for p in peaks:
            p.left, p.right = dsputils.peak_bounds(filtered, p, 0.01)
            p.area = np.sum(unfiltered[p.left:p.right+1])
        # Deal with overlapping peaks, store
        peaks = dsputils.remove_overlapping_peaks(peaks)
        event.peaks.extend(peaks)
        self.log.debug("Peakfinder found %s peaks." % len(event.peaks))
        return event

