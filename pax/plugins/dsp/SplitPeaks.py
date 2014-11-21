# Experimental peak splitting plugin

import math

import numpy as np

from pax import plugin, datastructure, dsputils


#TODO: move to separate module
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
        # TODO: this works on all peaks, but takes tpc and tpc_s2 as signals...
        filtered = event.get_waveform('tpc_s2').samples
        unfiltered = event.get_waveform('tpc').samples
        revised_peaks = []
        for parent in event.peaks:
            # If the peak is not large enough, it will not be split
            if ('composite_peak_min_width' in self.config and
                        parent.right - parent.left < self.config['composite_peak_min_width']
                    ):
                revised_peaks.append(parent)
                continue
            # Try to split the peak
            ps, vs = self.peaks_and_valleys(
                filtered[parent.left:parent.right + 1],
                test_function=self.is_valid_p_v_pair,
                # From Xerawdp:
                derivative_kernel=[-0.003059, -0.035187, -0.118739, -0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059]
            )
            # If the peak wasn't split, we don't have to do anything
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
            self.log.debug("S2 at " + str(parent.index_of_maximum) + ": peaks " + str(ps) + ", valleys " + str(vs))
            # Compute basic quantities for the sub-peaks
            for i, p in enumerate(ps):
                l_bound = vs[i - 1] if i != 0 else parent.left
                r_bound = vs[i]
                max_idx = l_bound + np.argmax(unfiltered[l_bound:r_bound + 1])
                new_peak = datastructure.Peak({
                    'index_of_maximum': max_idx,
                    'height':           unfiltered[max_idx],
                })
                # No need to recompute peak bounds: the whole parent peak is <0.01 max of the biggest peak
                # If we ever need to anyway, this code works:
                # left, right = dsputils.peak_bounds(filtered[l_bound:r_bound+1], max_i, 0.01)
                # new_peak.left  = left + l_bound
                # new_peak.right = right + l_bound
                new_peak.left = l_bound
                new_peak.right = r_bound
                revised_peaks.append(new_peak)
                new_peak.area = np.sum(unfiltered[new_peak.left:new_peak.right + 1])

        event.peaks = revised_peaks
        return event

    @staticmethod
    def peaks_and_valleys(signal, test_function, derivative_kernel):
        """Find peaks and valleys based on derivative sign changes
        :param signal: signal to search in
        :param test_function: Function which accepts three args:
                - signal, signal begin tested
                - peak, index of peak
                - valley, index of valley
            must return True if peak/valley pair is acceptable, else False
        :return: two sorted lists: peaks, valleys
        """
        assert len(derivative_kernel) % 2 == 1
        if len(signal) < len(derivative_kernel):
            # Signal is too small, can't calculate derivatives
            return [], []
        slope = np.convolve(signal, derivative_kernel, mode='same')
        # Chop the invalid parts off - easier than mode='valid' and adding offset
        # to results
        offset = (len(derivative_kernel) - 1) / 2
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
