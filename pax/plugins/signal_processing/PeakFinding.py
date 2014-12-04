import numpy as np
import math
from pax import plugin, datastructure, dsputils
import pandas

# Used for diagnostic plotting only
# TODO: factor out to separate plugin
import matplotlib.pyplot as plt
import os

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
                        unfiltered_max_idx = offset + np.argmax(unfiltered_wave[offset : offset - left + right + 1])

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





# TODO: do low-E peakfinding in veto even if we see a high-E peak in TPC (and vice versa, although less important)
# Needs more proper separation of veto & tpc peaks than just ignore_previous_peaks option in peakfinder
class FindSmallPeaks(plugin.TransformPlugin):

    def startup(self):
        self.min_sigma = self.config['peak_minimum_sigma']
        self.initial_noise_sigma = self.config['noise_sigma_guess']
        # Optional settings
        self.filter_to_use = self.config.get('filter_to_use', None)
        self.give_up_after = self.config.get('give_up_after_peak_of_size', float('inf'))
        self.look_in_large_peak_tails = self.config.get('also_look_in_tails_of_large_peaks', False)
        self.make_diagnostic_plots_in = self.config.get('make_diagnostic_plots_in', None)
        if self.make_diagnostic_plots_in is not None:
            if not os.path.exists(self.make_diagnostic_plots_in):
                os.makedirs(self.make_diagnostic_plots_in)

    def transform_event(self, event):
        # ocs is shorthand for occurrences

        # Convert occurrences to pandas dataframe
        # TODO: iterating over a pandas frame is extremely slow, we shouldn't use it

        ocs_flat = []
        for channel, channel_occurrences in event.occurrences.items():
            for start_index, occurrence_waveform in channel_occurrences:
                end_index = start_index + len(occurrence_waveform) - 1
                ocs_flat.append({
                    'channel' : channel,
                    'start_index' : start_index,
                    'end_index' : end_index,
                    'n_peaks' : float('nan'),   # Will be set later
                })
        all_ocs = pandas.DataFrame(ocs_flat)

        # Any ultra-large peaks after which we can give up?
        large_peak_start_points = [p.left for p in event.peaks if p.area > self.give_up_after]
        if len(large_peak_start_points) > 0:
            give_up_after = min(large_peak_start_points)
        else:
            give_up_after = float('inf')

        # Find regions free of peaks (found previously by the large peak finder)
        # Then search for small peaks in each of these regions
        for region_left, region_right in dsputils.free_regions(event):

            # Can we give up yet?
            if region_left >= give_up_after:
                self.log.debug("Giving up small-peak finding due to a peak > %s pe starting at %s" %
                               (self.give_up_after, give_up_after))
                break

            # Determine which occurrences to search in
            if self.look_in_large_peak_tails:
                # Which occurrences are partially in the free region?
                ocs = all_ocs[(all_ocs.start_index < region_right) & (all_ocs.end_index > region_left)]
            else:
                # Which occurrences are completely in the free region?
                ocs = all_ocs[(all_ocs.start_index >= region_left) & (all_ocs.end_index <= region_right)]

            if len(ocs) == 0:
                continue
            self.log.debug("Free region %s-%s: process %s occurrences" % (region_left, region_right, len(ocs)))


            for index, oc in ocs.iterrows():

                # In case we search occurrences only partially in free regions, we need to slice them
                if self.look_in_large_peak_tails:
                    # subtract oc['start_index'] to give an index in the occurrence waveform
                    start = max(region_left,  oc['start_index'])
                    stop  = min(region_right, oc['end_index'])
                else:
                    # No need to slice the occurrence
                    start = oc['start_index']
                    stop = oc['end_index']

                # Retrieve the waveform from pmt_waveforms
                # Do this only here: no sense retrieiving it for occurrences you are not going to test
                w = event.pmt_waveforms[oc['channel'], start:stop +1]
                origw = w

                # Apply the filter, if needed
                if self.filter_to_use is not None:
                    w = np.convolve(w, self.filter_to_use, 'same')

                # Use three passes to separate noise / peaks, see description in .... TODO
                # TODO: this is quite an expensive inner loop: maybe move to C? Or am I missing a clever speedup?
                noise_sigma = self.initial_noise_sigma
                for pass_number in range(3):
                    # TODO: use sliding window integral over 2 samples
                    raw_peaks = self.find_peaks(w, noise_sigma)
                    if pass_number != 0 and raw_peaks == old_raw_peaks:
                        # No change in peakfinding, previous noise level is still valid
                        # That means there's no point in repeating peak finding either, and we can just:
                        break
                    noise_sigma = w[self.samples_without_peaks(w, raw_peaks)].std()
                    old_raw_peaks = raw_peaks
                    # You can't break if you find no peaks: maybe the estimated noise level was too high

                # TODO: move to separate plugin
                if self.make_diagnostic_plots_in is not None:
                    plt.figure()
                    if self.filter_to_use is None:
                        plt.plot(w, drawstyle='steps', label='data')
                    else:
                        plt.plot(w, drawstyle='steps', label='data (filtered)')
                        plt.plot(origw, drawstyle='steps', label='data (raw)')
                    for p in raw_peaks:
                        plt.axvspan(p[0]-1, p[2], color='red', alpha=0.5)
                    plt.plot(noise_sigma * np.ones(len(w)), '--', label='1 sigma')
                    plt.plot(self.min_sigma * noise_sigma * np.ones(len(w)), '--', label='%s sigma' % self.min_sigma)
                    plt.legend()
                    bla = (event.event_number, oc['start_index'], oc['end_index'], oc['channel'],)
                    plt.title('Event %s, occurrence %d-%d, Channel %d' % bla)
                    plt.savefig(os.path.join(self.make_diagnostic_plots_in,  'event%04d_occ%06d-%06d_ch%03d.png' % bla))
                    plt.close()

                # Store the found peaks in the datastructure
                # ocs.loc[index,'n_peaks'] = len(raw_peaks)
                event.channel_peaks.extend([datastructure.ChannelPeak({
                    # TODO: store occurrence index -- occurrences needs to be a better datastructure first
                    'channel':             oc['channel'],
                    'left':                start + p[0],
                    'index_of_maximum':    start + p[1],
                    'right':               start + p[2],
                    'area':                np.sum(w[p[0]:p[2]+1]),
                    'height':              w[p[1]],
                    'noise_sigma':         noise_sigma,
                }) for p in raw_peaks])

        return event


    def find_peaks(self, w, noise_sigma):
        """
        Find all peaks at least self.min_sigma * noise_sigma above baseline.
        Peak boundaries are last samples above noise_sigma
        :param w: waveform to check for peaks
        :param noise_sigma: stdev of the noise
        :return: list of (left_index, max_index, right_index) tuples
        """
        peaks = []

        for left, right in dsputils.intervals_above_threshold(w, noise_sigma):
            max_idx = left + np.argmax(w[left:right + 1])
            height = w[max_idx]
            if height < noise_sigma * self.min_sigma:
                continue
            peaks.append((left, max_idx, right))

        return peaks

    def samples_without_peaks(self, w, peaks):
        not_in_peak = np.ones(len(w), dtype=np.bool)    # All True
        for p in peaks:
            not_in_peak[p[0]:p[1] + 1] = False
        return not_in_peak

