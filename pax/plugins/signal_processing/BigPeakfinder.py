import numpy as np
import math
from pax import plugin, datastructure, utils


class FindBigPeaks(plugin.TransformPlugin):

    """Find peaks in intervals above threshold in the sum waveform
    This type of peakfinding is only reliable for peaks >> 1pe
    (unless you have no noise, or use a battery of peak shape & isolation tests)

    If the low-energy peakfinder is good, this can be drastically simplified,
    e.g. we could just do one pass at the filtered waveform, large S1s should trigger this too.
    """

    def startup(self):
        self.derivative_kernel = self.config['derivative_kernel']

    def transform_event(self, event):
        for pf in self.config['peakfinders']:
            pfwave_obj = event.get_sum_waveform(pf['peakfinding_wave'])
            peakfinding_wave = pfwave_obj.samples
            detector = pfwave_obj.detector
            unfiltered_wave = event.get_sum_waveform(pf['unfiltered_wave']).samples

            # Define the peak/valley tester for the peaksplitter this peakfinder
            def is_valid_p_v_pair(signal, peak, valley):
                return (
                    abs(peak - valley) >= pf.get('min_p_v_distance', 0) and
                    signal[peak] / signal[valley] >= pf.get('min_p_v_ratio', 0) and
                    signal[peak] - signal[valley] >= pf.get('min_p_v_difference', 0)
                )

            # Search for peaks in the free regions
            for region_left, region_right in utils.free_regions(event, detector):
                for itv_left, itv_right in utils.intervals_where(
                        peakfinding_wave[region_left:region_right + 1] > pf['threshold']):

                    peaks = []
                    peak_wave = peakfinding_wave[itv_left:itv_right + 1]
                    if len(peak_wave) < pf.get('min_interval_width', 0):
                        continue

                    # We've found an interval above threshold: should we split it?
                    if len(peak_wave) < pf.get('min_split_attempt_width', float('inf')):
                        # No, the interval is too small
                        peaks.append((0, len(peak_wave) - 1))
                    else:
                        # Yes, try it
                        peak_indices, valley_indices = self.peaks_and_valleys(peak_wave,
                                                                              test_function=is_valid_p_v_pair)
                        if len(peak_indices) <= 1:
                            # The peak was not split
                            peaks.append((0, len(peak_wave) - 1))
                        else:
                            # It was split, add the subpeaks
                            for i, _ in enumerate(peak_indices):
                                left = valley_indices[i - 1] if i != 0 else 0
                                right = valley_indices[i] if i != len(peak_indices) else len(peak_wave) - 1
                                peaks.append((left, right))
                            # Show peak splitting diagnostic plot
                            # TODO: Make option, or factor to separate plugin
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
                        offset = region_left + itv_left + left
                        unfiltered_max_idx = offset + np.argmax(unfiltered_wave[offset: offset - left + right + 1])

                        event.peaks.append(datastructure.Peak({
                            'index_of_maximum': unfiltered_max_idx,   # in unfiltered waveform!
                            'height':           unfiltered_wave[unfiltered_max_idx],
                            'left':             region_left + itv_left + left,
                            'right':            region_left + itv_left + right,
                            'detector':         detector}))

        return event

    def peaks_and_valleys(self, signal, test_function):
        """Find peaks and valleys based on derivative sign changes
        :param signal: signal to search in
        :param test_function: Function which accepts three args:
                - signal, signal begin tested
                - peak, index of peak
                - valley, index of valley
            must return True if peak/valley pair is acceptable, else False
        :return: two sorted numpy arrays: peaks, valleys
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

        peaks, valleys = utils.where_changes(slope > 0)
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
            # The peak is not split
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

        peaks, valleys = [p for p in peaks if not math.isnan(p)], [v for v in valleys if not math.isnan(v)]

        # Return all remaining peaks & valleys
        return np.array(peaks), np.array(valleys)
