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

        # Initialize empty waveforms
        pmts = 1 + max(self.config['pmts_veto'])   # TODO: really??
        event.pmt_waveforms = np.zeros((pmts, event.length()))
        for group, members in self.channel_groups.items():
            event.waveforms.append(datastructure.Waveform({
                'samples':  np.zeros(event.length()),
                'name':     group,
                'pmt_list': self.crazy_type_conversion(members),
            }))
        # TODO: Should these go into event class?
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
                end_index = start_index + len(pulse_wave)   # Well, not really index... index+1
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
        #filter_sigma = self.config['gauss_filter_fwhm']/2.35482
        #x = np.linspace(-2,2,num=(1+4*filter_sigma/self.config['digitizer_t_resolution']))
        #self.filter_ir = 1/(filter_sigma*np.sqrt(2*np.pi)) * np.exp(-x**2 / (2*filter_sigma**2))
        # Guillaume's raised cosine coeffs:
        self.filter_ir = [0.005452,  0.009142,  0.013074,  0.017179,  0.021381,  0.025597,  0.029746,  0.033740,  0.037499,  0.040941,  0.043992,  0.046586,  0.048666,  0.050185,  0.051111,
                          0.051422,  0.051111,  0.050185,  0.048666,  0.046586,  0.043992,  0.040941,  0.037499,  0.033740,  0.029746,  0.025597,  0.021381,  0.017179,  0.013074,  0.009142,  0.005452]
        self.filter_ir /= np.sum(self.filter_ir)

    def transform_event(self, event):
        input_w = event.get_waveform('tpc')
        event.waveforms.append(datastructure.Waveform({
            'name':     'filtered_for_s2',
            'samples':   #np.convolve(
                np.convolve(input_w.samples, self.filter_ir, 'same'),
                #self.filter_ir, 'same'
            #),
            'pmt_list':  input_w.pmt_list,
        }))

        return event


class FindS2s(plugin.TransformPlugin):

    def transform_event(self, event):
        filtered = event.get_waveform('filtered_for_s2').samples
        unfiltered = event.get_waveform('tpc').samples
        # Find intervals above threshold in filtered waveform
        candidate_intervals = dsputils.intervals_above_threshold(filtered, self.config['s2_threshold'])
        # Find peaks in intervals
        peaks = self.find_peaks_in_intervals(unfiltered, candidate_intervals)
        # Compute peak extents - on FILTERED waveform!
        for p in peaks:
            p.left, p.right = dsputils.peak_bounds(filtered, p.index_of_maximum, 0.01)
            p.area = np.sum(unfiltered[p.left:p.right+1])
        # Merge overlapping peaks, we'll split them later
        peaks = dsputils.merge_overlapping_peaks(peaks)
        event.peaks.extend(peaks)
        self.log.debug("Found %s peaks." % len(event.peaks))
        return event

    @staticmethod
    def find_peaks_in_intervals(signal, candidate_intervals):
        peaks = []
        for itv_left, itv_right in candidate_intervals:
            max_idx = itv_left + np.argmax(signal[itv_left:itv_right + 1])
            peaks.append(datastructure.Peak({
                    'index_of_maximum': max_idx,
                    'height':           signal[max_idx],
            }))
        return peaks


class SplitPeaks(plugin.TransformPlugin):

    def startup(self):
        def is_valid_p_v_pair(signal, peak, valley):
            return (
                abs(peak - valley) >= self.config['min_p_v_distance'] and
                signal[peak] / signal[valley] >= self.config['min_p_v_ratio'] and
                signal[peak] - signal[valley] >= self.config['min_p_v_difference']
            )
        self.is_valid_p_v_pair = is_valid_p_v_pair


    def transform_event(self, event):
        filtered = event.get_waveform('filtered_for_s2').samples
        unfiltered = event.get_waveform('tpc').samples
        revised_peaks = []
        for parent in event.peaks:
            ps, vs = dsputils.peaks_and_valleys(
                filtered[parent.left:parent.right+1],
                test_function=self.is_valid_p_v_pair
            )
            # If the peak isn't composite, we don't have to do anything
            if len(ps) < 2:
                revised_peaks.append(parent)
                continue

            # import matplotlib.pyplot as plt
            # plt.plot(event.get_waveform('tpc').samples[parent.left:parent.right+1])
            # plt.plot(filtered[parent.left:parent.right+1])
            # plt.plot(ps, filtered[parent.left + np.array(ps)], 'or')
            # plt.plot(vs, filtered[parent.left + np.array(vs)], 'ob')
            # plt.show()

            ps += parent.left
            vs += parent.left
            self.log.debug("S2 at "+ str(parent.index_of_maximum) +": peaks " + str(ps) + ", valleys "+str(vs))
            # Compute basic quantities for the sub-peaks
            for i, p in enumerate(ps):
                l_bound = vs[i-1] if i!=0 else parent.left
                r_bound = vs[i]
                max_idx = l_bound + np.argmax(unfiltered[l_bound:r_bound+1])
                new_peak = datastructure.Peak({
                        'index_of_maximum': max_idx,
                        'height':           unfiltered[max_idx],
                })
                # No need to recompute peak bounds: the whole parent peak is <0.01 max of the biggest peak
                # If we ever need to anyway, this code works:
                # left, right = dsputils.peak_bounds(filtered[l_bound:r_bound+1], max_idx - l_bound, 0.01)
                # new_peak.left  = left + l_bound
                # new_peak.right = right + l_bound
                new_peak.left = l_bound
                new_peak.right = r_bound
                revised_peaks.append(new_peak)
                new_peak.area = np.sum(unfiltered[new_peak.left:new_peak.right+1])

        event.peaks = revised_peaks
        return event

class IdentifyPeaks(plugin.TransformPlugin):
    def transform_event(self, event):
        #PLACEHOLDER:
        # if area in 5 samples around max i s > 50% of total area, christen as S1
        unfiltered = event.get_waveform('tpc').samples
        for p in event.peaks:
            if np.sum(unfiltered[p.index_of_maximum -2: p.index_of_maximum + 3]) > 0.5*p.area:
                p.type = 's1'
                self.log.debug("%s-%s-%s: S1" % (p.left, p.index_of_maximum, p.right))
            else:
                p.type = 's2'
                self.log.debug("%s-%s-%s: S2" % (p.left, p.index_of_maximum, p.right))
        return event
