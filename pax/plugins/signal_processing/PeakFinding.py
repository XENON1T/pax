import numpy as np
import math
from pax import plugin, datastructure, dsputils

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
                for itv_left, itv_right in dsputils.intervals_where(
                        peakfinding_wave[region_left:region_right + 1] > pf['threshold']):

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
        peaks, valleys = dsputils.where_changes(slope > 0)
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

        # Get settings from configuration
        self.min_sigma = self.config['peak_minimum_sigma']
        self.initial_noise_sigma = self.config['noise_sigma_guess']

        # Optional settings
        self.filter_to_use = self.config.get('filter_to_use', None)
        self.give_up_after = self.config.get('give_up_after_peak_of_size', float('inf'))
        self.max_noise_detection_passes = self.config.get('max_noise_detection_passes', float('inf'))
        self.make_diagnostic_plots_in = self.config.get('make_diagnostic_plots_in', None)
        if self.make_diagnostic_plots_in is not None:
            if not os.path.exists(self.make_diagnostic_plots_in):
                os.makedirs(self.make_diagnostic_plots_in)

    def transform_event(self, event):
        # ocs is shorthand for occurrences, as usual

        # Any ultra-large peaks after which we can give up?
        large_peak_start_points = [p.left for p in event.peaks if p.area > self.give_up_after]
        if len(large_peak_start_points) > 0:
            give_up_after = min(large_peak_start_points)
        else:
            give_up_after = float('inf')

        noise_count = {}
        event.bad_channels = []

        # Get all free regions before the give_up_after point
        for region_left, region_right in dsputils.free_regions(event):

            # Can we give up yet?
            if region_left >= give_up_after:
                break

            # Find all occurrences completely enveloped in the free region. Thank pyintervaltree for log(n) run
            # In the future we may enable strict=False, so we're not relying on the zero-suppression to separate
            # small peaks close to a large peak
            ocs = event.occurrences_interval_tree.search(region_left, region_right, strict=True)
            self.log.debug("Free region %05d-%05d: process %s occurrences" % (region_left, region_right, len(ocs)))

            for oc in ocs:
                # Focus only on the part of the occurrence inside the free region (superfluous as long as strict=True)
                # Remember: intervaltree uses half-open intervals, stop is the first index outside
                start = max(region_left, oc.begin)
                stop = min(region_right + 1, oc.end)
                channel = oc.data['channel']

                # Maybe some channels have already been marked as bad (configuration?), don't consider these.
                if channel in event.bad_channels:
                    continue

                # Retrieve the waveform from pmt_waveforms
                w = event.pmt_waveforms[channel, start:stop]

                # Keep a copy, so we can filter w if needed:
                origw = w

                # Apply the filter, if user wants to
                if self.filter_to_use is not None:
                    w = np.convolve(w, self.filter_to_use, 'same')

                # Use three passes to separate noise / peaks, see description in .... TODO
                noise_sigma = self.initial_noise_sigma
                old_raw_peaks = []
                pass_number = 0
                while True:
                    # Determine the peaks based on the noise level
                    # Can't just use w > self.min_sigma * noise_sigma here, want to extend peak bounds to noise_sigma
                    raw_peaks = self.find_peaks(w, noise_sigma)

                    if pass_number != 0 and raw_peaks == old_raw_peaks:
                        # No change in peakfinding, previous noise level is still valid
                        # That means there's no point in repeating peak finding either, and we can just:
                        break
                        # This saves about 25% of runtime
                        # You can't break if you find no peaks on the first pass:
                        # maybe the estimated noise level was too high

                    # Correct the baseline -- BuildWaveforms can get it wrong if there is a pe in the starting samples
                    w -= w[self.samples_without_peaks(w, raw_peaks)].mean()

                    # Determine the new noise_sigma
                    noise_sigma = w[self.samples_without_peaks(w, raw_peaks)].std()

                    old_raw_peaks = raw_peaks
                    if pass_number >= self.max_noise_detection_passes:
                        self.log.warning((
                            "In occurrence %s-%s in channel %s, findSmallPeaks did not converge on peaks after %s" +
                            " iterations. This could indicate a baseline problem in this occurrence. " +
                            "Channel-based peakfinding in this occurrence may be less accurate.") % (
                                start, stop, channel, pass_number))
                        break

                    pass_number += 1

                # Update the noise occurrence count
                if len(raw_peaks) == 0:
                    noise_count[channel] = noise_count.get(channel, 0) + 1

                # Store the found peaks in the datastructure
                peaks = []
                for p in raw_peaks:
                    peaks.append(datastructure.ChannelPeak({
                        # TODO: store occurrence index -- occurrences needs to be a better datastructure first
                        'channel':             channel,
                        'left':                start + p[0],
                        'index_of_maximum':    start + p[1],
                        'right':               start + p[2],
                        # NB: area and max are computed in filtered waveform, because
                        # the sliding window filter will shift the peak shape a bit
                        'area':                np.sum(w[p[0]:p[2]+1]),
                        'height':              w[p[1]],
                        'noise_sigma':         noise_sigma,
                    }))
                event.channel_peaks.extend(peaks)

                # TODO: move to separate plugin?
                if self.make_diagnostic_plots_in:
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
                    bla = (event.event_number, start, stop, channel)
                    plt.title('Event %s, occurrence %d-%d, Channel %d' % bla)
                    plt.savefig(os.path.join(self.make_diagnostic_plots_in,  'event%04d_occ%05d-%05d_ch%03d.png' % bla))
                    plt.close()

        # Mark channels with an abnormally high noise rate as bad
        for ch, dc in noise_count.items():
            if dc > self.config['maximum_noise_occurrences_per_channel']:
                self.log.debug(
                    "Channel %s shows an abnormally high rate of noise pulses (%s): its spe pulses will be excluded" % (
                        ch, dc))
                event.bad_channels.append(ch)

        return event


    def find_peaks(self, w, noise_sigma):
        """
        Find all peaks at least self.min_sigma * noise_sigma above baseline.
        Peak boundaries are last samples above noise_sigma
        :param w: waveform to check for peaks
        :param noise_sigma: stdev of the noise
        :return: peaks as list of (left_index, max_index, right_index) tuples
        """
        peaks = []

        for left, right in dsputils.intervals_where(w > noise_sigma):
            max_idx = left + np.argmax(w[left:right + 1])
            height = w[max_idx]
            if height < noise_sigma * self.min_sigma:
                continue
            peaks.append((left, max_idx, right))
        return peaks

    def samples_without_peaks(self, w, peaks):
        """Return array of bools of same size as w, True if none of peaks live there"""
        not_in_peak = np.ones(len(w), dtype=np.bool)    # All True
        for p in peaks:
            not_in_peak[p[0]:p[2] + 1] = False
        return not_in_peak

