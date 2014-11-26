import numpy as np
import math
from pax import plugin, datastructure, dsputils


class FindBigPeaks(plugin.TransformPlugin):
    """Find peaks in intervals above threshold in the sum waveform
    This type of peakfinding is only reliable for peaks >> 1pe
    (unless you have no noise, or use a battery of peak shape & isolation tests)

    If the low-energy peakfinder is good, this can be drastically simplified,
    e.g. we could just do one pass at the filtered waveform, large S1s should trigger this too.
    """

    def startup(self):
        self.derivative_kernel=[-0.003059, -0.035187, -0.118739, -0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059]
        #TODO: put in config

    def transform_event(self, event):
        for pf in self.config['peakfinders']:
            peakfinding_wave = event.get_waveform(pf['peakfinding_wave']).samples
            unfiltered_wave = event.get_waveform(pf['unfiltered_wave']).samples
            peaks = []

            # Define the peak/valley tester for this peakfinder
            def is_valid_p_v_pair( signal, peak, valley):
                return (
                    abs(peak - valley) >= pf.get('min_p_v_distance', 0) and
                    signal[peak] / signal[valley] >= pf.get('min_p_v_ratio', 0) and
                    signal[peak] - signal[valley] >= pf.get('min_p_v_difference', 0)
                )

            # Find regions currently free of peaks
            if len(event.peaks) == 0 or pf.get('ignore_previous_peaks', False):
                pf_regions = [(0, len(peakfinding_wave) - 1)]
            else:
                pf_regions = dsputils.free_regions(event)

            # Search for peaks in the free regions
            for region_left, region_right in pf_regions:
                for itv_left, itv_right in dsputils.intervals_above_threshold(
                        peakfinding_wave[region_left:region_right + 1], pf['threshold']):

                    peaks = []
                    peak_wave = peakfinding_wave[itv_left:itv_right + 1]
                    if len(peak_wave) < pf.get('min_interval_width', 0):
                        continue

                    if np.sum(peak_wave) < pf.get('area_threshold', 0):
                        # Let the low-energy peakfinder take care of this one
                        continue

                    # We've found an interval above threshold: should we split it?
                    if len(peak_wave) < pf.get('min_split_attempt_width', float('inf')):
                        # No, the interval is too small
                        peaks.append((0,len(peak_wave)-1))
                    else:
                        # Yes, try it
                        peak_indices, valley_indices = self.peaks_and_valleys(peak_wave,
                                                                              test_function=is_valid_p_v_pair)
                        if len(peak_indices) <= 1:
                            # The peak was not split
                            peaks.append((0,len(peak_wave)-1))
                        else:
                            # It was split, add the subpeaks
                            for i, _ in enumerate(peak_indices):
                                left = valley_indices[i - 1] if i != 0 else 0
                                right = valley_indices[i] if i != len(peak_indices) else len(peak_wave)-1
                                peaks.append((left, right))
                            # Show peak splitting diagnostic plot
                            # import matplotlib.pyplot as plt
                            # plt.plot(1+peak_wave)
                            # plt.plot(peak_indices,   1+peak_wave[np.array(peak_indices)], 'or')
                            # plt.plot(valley_indices, 1+peak_wave[np.array(valley_indices)], 'ob')
                            # plt.yscale('log')
                            # plt.ylabel('1+amplitude (pe)')
                            # plt.xlabel('time (digitizer bin = 10ns)')
                            # plt.show()

                    # Add all the found peaks to the event
                    for left, right in peaks:
                        unfiltered_max_idx = region_left + itv_left + left + np.argmax(peak_wave[left:right+1])

                        event.peaks.append(datastructure.Peak({
                            'index_of_maximum': unfiltered_max_idx,   # in unfiltered waveform!
                            'height':           unfiltered_wave[unfiltered_max_idx],
                            'left':             region_left + itv_left + left,
                            'right':            region_left + itv_left + right}))

                        # Should we already label the peak?
                        if 'force_peak_label' in pf:
                            event.peaks[-1].type = pf['force_peak_label']

            self.log.debug("Found %s peaks in %s." % (len(peaks), pf['peakfinding_wave']))

        return event


    def peaks_and_valleys(self, signal, test_function):
        """Find peaks and valleys based on derivative sign changes
        :param signal: signal to search in
        :param test_function: Function which accepts three args:
                - signal, signal begin tested
                - peak, index of peak
                - valley, index of valley
            must return True if peak/valley pair is acceptable, else False
        :return: two sorted lists: peaks, valleys
        The peaks always occur before the valleys.
        """
        assert len(self.derivative_kernel) % 2 == 1
        if len(signal) < len(self.derivative_kernel):
            # Signal is too small, can't calculate derivatives
            return [], []
        slope = np.convolve(signal, self.derivative_kernel, mode='same')
        # Chop the invalid parts off - easier than mode='valid' and adding offset
        # to results
        offset = (len(self.derivative_kernel) - 1) / 2
        slope[0:offset] = np.zeros(offset)
        slope[len(slope) - offset:] = np.zeros(offset)
        peaks, valleys = dsputils.sign_changes(slope, report_first_index='never')
        peaks = np.array(sorted(peaks))
        valleys = np.array(sorted(valleys))
        assert len(peaks) == len(valleys)
        # Remove coinciding peak&valleys
        good_indices = np.where(peaks != valleys)[0]
        peaks = np.array(peaks[good_indices])
        valleys = np.array(valleys[good_indices])
        if not np.all(valleys > peaks):   # Valleys are AFTER the peaks
            print(valleys - peaks)
            raise RuntimeError("Peak & valley list weird!")

        if len(peaks) < 2:
            return peaks, valleys

        # Remove peaks and valleys which are too close to each other, or have too low a p/v ratio
        # This can't be a for-loop, as we are modifying the lists, and step back
        # to recheck peaks.
        now_at_peak = 0
        while 1:

            # Find the next peak, if there is one
            if now_at_peak > len(peaks) - 1:
                break
            peak = peaks[now_at_peak]
            if math.isnan(peak):
                now_at_peak += 1
                continue

            # Check the valleys around this peak
            if peak < min(valleys):
                fail_left = False
            else:
                valley_left = np.max(valleys[np.where(valleys < peak)[0]])
                fail_left = not test_function(signal, peak, valley_left)
            valley_right = np.min(valleys[np.where(valleys > peak)[0]])
            fail_right = not test_function(signal, peak, valley_right)
            if not (fail_left or fail_right):
                # We're good, move along
                now_at_peak += 1
                continue

            # Some check failed: we must remove a peak/valley pair.
            # Which valley should we remove?
            if fail_left and fail_right:
                # Both valleys are bad! Remove the most shallow valley.
                valley_to_remove = valley_left if signal[
                    valley_left] > signal[valley_right] else valley_right
            elif fail_left:
                valley_to_remove = valley_left
            elif fail_right:
                valley_to_remove = valley_right

            # Remove the shallowest peak near the valley marked for removal
            left_peak = max(peaks[np.where(peaks < valley_to_remove)[0]])
            if valley_to_remove > max(peaks):
                # There is no right peak, so remove the left peak
                peaks = peaks[np.where(peaks != left_peak)[0]]
            else:
                right_peak = min(peaks[np.where(peaks > valley_to_remove)[0]])
                if signal[left_peak] < signal[right_peak]:
                    peaks = peaks[np.where(peaks != left_peak)[0]]
                else:
                    peaks = peaks[np.where(peaks != right_peak)[0]]

            # Jump back a few peaks to be sure we repeat all checks,
            # even if we just removed a peak before the current peak
            now_at_peak = max(0, now_at_peak - 1)
            valleys = valleys[np.where(valleys != valley_to_remove)[0]]

        peaks, valleys = [p for p in peaks if not math.isnan(
            p)], [v for v in valleys if not math.isnan(v)]
        # Return all remaining peaks & valleys
        return np.array(peaks), np.array(valleys)


# class PruneSmallPeaksNearLargePeaks(plugin.TransformPlugin):
#     """Delete small peaks too close to a large peak
#     Large peaks turn on many channels so the noise in the sum waveform noise becomes large
#     (if the noise is gaussian, it scales as ~sqrt(nchannels).
#     The low-energy peakfinder should take care of fake peaks in these regions just fine,
#     but it is expensive, so if we are certain a peak can't be real, might as well delete it already.
#     """
#
#     def transform_event(self, event):
#         large_peaks = [p for p in event.peaks if p.area >= self.config['large_peaks_start_from']]
#         small_peaks = [p for p in event.peaks if p.area < self.config['always_keep_peaks_larger_than']]
#         print(large_peaks)
#         for p in small_peaks:
#             if p.type in self.config['never_prune_peak_types']:
#                 continue
#
#             largepeakstoleft = [q for q in large_peaks if q.left < p.left]
#             if len(largepeakstoleft) > 0:
#                 leftlargepeak = max(largepeakstoleft, key=lambda x : x.left)
#                 if p.left - leftlargepeak.right < self.config['min_distance_to_large_peak']:
#                     self.log.warning("Prune %s at %s-%s-%s: too close to large peak on the left" % (
#                         p.type, p.left, p.index_of_maximum, p.right))
#                     p.type = 'X'
#                     continue
#
#             largepeakstoright = [q for q in large_peaks if q.right > p.right]
#             if len(largepeakstoright) > 0:
#                 rightlargepeak = min(largepeakstoright, key=lambda x : x.right)
#                 if rightlargepeak.left - p.right < self.config['min_distance_to_large_peak']:
#                     self.log.warning("Prune %s at %s-%s-%s: too close to large peak on the right" % (
#                         p.type, p.left, p.index_of_maximum, p.right))
#                     p.type = 'X'
#                     continue
#
#         #event.peaks = [p for p in event.peaks if p.type != 'X']
#         return event



class ComputePeakWidths(plugin.TransformPlugin):
    """Does what it says on the tin"""

    def transform_event(self, event):
        for peak in event.peaks:

            # Check if peak is sane
            if peak.index_of_maximum < peak.left:
                self.log.debug("Insane peak %s-%s-%s, can't compute widths!" % (
                    peak.left, peak.index_of_maximum, peak.right))
                continue

            for width_name, conf in self.config['width_computations'].items():

                peak[width_name] = dsputils.width_at_fraction(
                    peak_wave=event.get_waveform(conf['waveform_to_use']).samples[peak.left : peak.right+1],
                    fraction_of_max=conf['fraction_of_max'],
                    max_idx=peak.index_of_maximum - peak.left,
                    interpolate=conf['interpolate'])

        return event




class ComputePeakAreas(plugin.TransformPlugin):

    def transform_event(self, event):
        for peak in event.peaks:

            # Compute area in each channel
            peak.area_per_pmt = np.sum(event.pmt_waveforms[:, peak.left:peak.right+1], axis=1)

            # Determine which channels contribute to the peak's total area
            peak.contributing_pmts = np.array(
                np.where(peak.area_per_pmt >= self.config['minimum_area'])[0],
                dtype=np.uint16)

            # Compute the peak's areas
            # TODO: make excluding non-contributing pmts optional
            if peak.type == 'veto':
                peak.area = np.sum(peak.area_per_pmt[list(self.config['pmts_veto'])])
            else:
                if self.config['exclude_non_contributing_channels_from_area']:
                    peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])
                else:
                    peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])

        return event


class ComputePeakEntropies(plugin.TransformPlugin):
    #TODO: write tests


    def transform_event(self, event):
        for peak in event.peaks:

            peak_waveforms = event.pmt_waveforms[:, peak.left:peak.right+1]

            # Switching from entropy to kurtosis doesn't make it faster...
            # At head put:
            # import scipy
            # from scipy import stats
            # Here put:
            # peak.entropy_per_pmt = scipy.stats.kurtosis(peak_waveforms, axis=1)
            # continue

            if self.config['normalization_mode'] is 'abs':
                normalized = np.abs(peak_waveforms)
            elif self.config['normalization_mode'] is 'square':
                normalized = peak_waveforms**2
            else:
                raise ValueError(
                    'Invalid Configuration for ComputePeakEntropies: normalization_mode must be abs or square')

            # In the case of abs, we could re-use peak.area_per_pmt to normalize
            # This gains only a little bit of performance, and 'square' is what we use in Xenon100 anyway.
            # Note the use of np.newaxis to enable numpy broadcasting of the division
            normalized /= peak.area_per_pmt[:, np.newaxis]

            if self.config['only_for_contributing_pmts']:
                # Could this be vectorized better?
                # There is probably little use in restricting to a set of pmts before here,
                # the logarithm contains most of the work.
                peak.entropy_per_pmt = np.zeros(len(peak_waveforms))
                for pmt in peak.contributing_pmts:
                    peak.entropy_per_pmt[pmt] = -np.sum(normalized[pmt]*np.log(normalized[pmt]))
            else:
                peak.entropy_per_pmt = -np.sum(normalized*np.log(normalized), axis=1)

        return event



class IdentifyPeaks(plugin.TransformPlugin):

    def transform_event(self, event):

        unfiltered = event.get_waveform('tpc').samples
        for p in event.peaks:
            if p.type != 'unknown':
                # Some peakfinder forced the type. Fine, not my problem...
                continue
            # PLACEHOLDER:
            # if area in s1_half_area_in samples around max is > 50% of total area, christen as S1 candidate
            # if peak is smaller than s1_half_area_in, it is certainly an s1
            if p.right - p.left < self.config['s1_half_area_in']:
                p.type = 's1'
            else:
                left_samples = math.floor(self.config['s1_half_area_in']/2)
                right_samples = math.ceil(self.config['s1_half_area_in']/2)
                if np.sum(unfiltered[p.index_of_maximum - left_samples: p.index_of_maximum + right_samples]) > 0.5 * p.area:
                    p.type = 's1'
                    #self.log.debug("%s-%s-%s: S1" % (p.left, p.index_of_maximum, p.right))
                else:
                    p.type = 's2'
                    #self.log.debug("%s-%s-%s: S2" % (p.left, p.index_of_maximum, p.right))
        return event
