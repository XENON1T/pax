import numpy as np
import numba

from pax import plugin, datastructure, utils


# Used for diagnostic plotting only
# TODO: factor out to separate plugin?
import matplotlib.pyplot as plt
import os


class FindSmallPeaks(plugin.TransformPlugin):

    def startup(self):

        # Get settings from configuration
        self.min_sigma = self.config['peak_minimum_sigma']
        self.initial_noise_sigma = self.config['noise_sigma_guess']

        # Optional settings
        self.filter_to_use = self.config.get('filter_to_use', None)
        self.give_up_after = self.config.get('give_up_after_peak_of_size', float('inf'))
        self.max_noise_detection_passes = self.config.get('max_noise_detection_passes', float('inf'))
        self.make_diagnostic_plots = self.config.get('make_diagnostic_plots', 'never')
        self.make_diagnostic_plots_in = self.config.get('make_diagnostic_plots_in', 'small_pf_diagnostic_plots')
        if self.make_diagnostic_plots != 'never':
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

        # Handle each detector separately
        for detector in self.config['channels_in_detector'].keys():
            self.log.debug("Finding channel peaks in data from %s" % detector)

            # Get all free regions before the give_up_after point
            for region_left, region_right in utils.free_regions(event, detector):

                # Can we give up yet?
                if region_left >= give_up_after:
                    break

                # Find all occurrences completely enveloped in the free region.
                # TODO: we should put strict=False, so we're not relying on the zero-suppression to separate
                # small peaks close to a large peak. However, right now this brings in stuff from large peaks if their
                # boundsare not completely tight...
                ocs = event.get_occurrences_between(region_left, region_right, strict=True)
                self.log.debug("Free region %05d-%05d: process %s occurrences" % (region_left, region_right, len(ocs)))

                for oc in ocs:
                    # Focus only on the part of the occurrence inside the free region
                    # (superfluous as long as strict=True)
                    start = max(region_left, oc.left)
                    stop = min(region_right, oc.right)
                    channel = oc.channel

                    # Don't consider channels from other detectors
                    if channel not in self.config['channels_in_detector'][detector]:
                        continue

                    # Don't consider dead channels
                    if self.config['gains'][channel] == 0:
                        continue

                    # Retrieve the waveform from channel_waveforms
                    w = event.channel_waveforms[channel, start: stop + 1]

                    # Keep the unfiltered waveform in origw
                    origw = w

                    # Apply the filter, if user wants to
                    if self.filter_to_use is not None:
                        w = np.convolve(w, self.filter_to_use, 'same')

                    # Use several passes to separate noise / peaks
                    noise_sigma = self.initial_noise_sigma
                    old_raw_peaks = []
                    pass_number = 0
                    baseline_correction_delta = 0
                    baseline_correction = 0
                    while True:
                        # Determine the peaks based on the noise level
                        # Can't just use w > self.min_sigma * noise_sigma here,
                        # want to extend peak bounds to noise_sigma
                        # TODO: will probably crash if more than 100 peaks in an occurrence found!!
                        raw_peaks = np.zeros((100, 2), dtype=np.int64)
                        n_raw_peaks_found = _numba_find_peaks(w,
                                                              noise_sigma,
                                                              self.min_sigma * noise_sigma,
                                                              raw_peaks)
                        raw_peaks = raw_peaks[:n_raw_peaks_found]

                        if pass_number != 0 and \
                                len(raw_peaks) == len(old_raw_peaks) and \
                                np.all(raw_peaks == old_raw_peaks):
                            # No change in peakfinding, previous noise level is still valid
                            # That means there's no point in repeating peak finding either, and we can just:
                            break
                            # You can't break if you find no peaks on the first pass:
                            # maybe the estimated noise level was too high

                        # Compute the baseline correction and the new noise_sigma
                        # -- BuildWaveforms can get baseline wrong if there is a pe in the starting samples
                        result = np.zeros(2)
                        mean_std_outside_peaks(w, raw_peaks, result)
                        baseline_correction_delta = result[0]
                        noise_sigma = result[1]

                        # Perform the baseline correction
                        w -= baseline_correction_delta
                        baseline_correction += baseline_correction_delta

                        old_raw_peaks = raw_peaks
                        if pass_number >= self.max_noise_detection_passes:
                            self.log.warning((
                                "In occurrence %s-%s in channel %s, findSmallPeaks did not converge on peaks after %s" +
                                " iterations. This could indicate a baseline problem in this occurrence. " +
                                "Channel-based peakfinding in this occurrence may be less accurate.") % (
                                    start, stop, channel, pass_number))
                            break

                        pass_number += 1

                    # Update occurrence data
                    oc.noise_sigma = noise_sigma
                    oc.baseline_correction = baseline_correction
                    oc.height = oc.height - baseline_correction

                    # Update the noise occurrence count
                    if len(raw_peaks) == 0:
                        noise_count[channel] = noise_count.get(channel, 0) + 1

                    # Store the found peaks in the datastructure
                    peaks = []
                    for p in raw_peaks:
                        # TODO: Hmzz.. height is computed in unfiltered waveform, noise_level in filtered!
                        # TODO: height is wrong if baseline is corrected...
                        # TODO: are you sure it is ok to compute area in filtered waveform?
                        max_idx = p[0] + np.argmax(origw[p[0]:p[1] + 1])
                        peaks.append(datastructure.ChannelPeak({
                            # TODO: store occurrence index
                            'channel':             channel,
                            'left':                start + p[0],
                            # Compute index_of_maximum in the original waveform: the filter introduces a phase shift
                            'index_of_maximum':    start + max_idx,
                            'right':               start + p[1],
                            # NB: area and max are computed in filtered waveform, because
                            # the sliding window filter will shift the peak shape a bit
                            'area':                np.sum(w[p[0]:p[1] + 1]),
                            'height':              origw[max_idx],
                            # Also store height in filtered waveform:
                            # useful to see what a higher threshold would have done
                            'filtered_height':     np.max(w[p[0]:p[1] + 1]),
                            'noise_sigma':         noise_sigma,
                        }))
                    event.all_channel_peaks.extend(peaks)

                    # TODO: move to separate plugin?
                    if self.make_diagnostic_plots == 'always' or \
                       self.make_diagnostic_plots == 'no peaks' and not len(peaks):
                        plt.figure()
                        if self.filter_to_use is None:
                            plt.plot(w, drawstyle='steps', label='data')
                        else:
                            plt.plot(w, drawstyle='steps', label='data (filtered)')
                            plt.plot(origw, drawstyle='steps', label='data (raw)', alpha=0.5, color='gray')
                        for p in raw_peaks:
                            plt.axvspan(p[0] - 1, p[1], color='red', alpha=0.5)
                        plt.plot(noise_sigma * np.ones(len(w)), '--', label='1 sigma')
                        plt.plot(self.min_sigma * noise_sigma * np.ones(len(w)),
                                 '--', label='%s sigma' % self.min_sigma)
                        plt.legend()
                        bla = (event.event_number, start, stop, channel)
                        plt.title('Event %s, occurrence %d-%d, Channel %d' % bla)
                        plt.savefig(os.path.join(self.make_diagnostic_plots_in,
                                                 'event%04d_occ%05d-%05d_ch%03d.png' % bla))
                        plt.close()

        # Mark channels with an abnormally high noise rate as bad
        for ch, dc in noise_count.items():
            if dc > self.config['maximum_noise_occurrences_per_channel']:
                self.log.debug(
                    "Channel %s shows an abnormally high rate of noise pulses (%s): its spe pulses will be excluded" % (
                        ch, dc))
                event.is_channel_bad[ch] = True

        return event


@numba.jit(numba.void(numba.float64[:], numba.int64[:, :], numba.float64[:]), nopython=True)
def mean_std_outside_peaks(w, raw_peaks, result):
    """Compute mean and std (rms) of samples w not in raw_peaks

    Both mean and std are computed in a single pass using a clever algorithm from:
    see http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

    :param w: waveform
    :param raw_peaks: np.int64[n,2]: raw peak boundaries
    :param result: np.array([0,0]) will contain result (mean, std) after evaluation
    :return: nothing
    """

    n = 0           # Samples outside peak included so far
    mean = 0        # Mean so far
    m2 = 0          # Sm of squares of differences from the (current) mean
    delta = 0       # Temp storage for x - mean
    n_peaks = len(raw_peaks)
    current_peak = 0
    currently_in_peak = False

    for i, x in enumerate(w):

        if currently_in_peak:
            if i > raw_peaks[current_peak, 1]:
                current_peak += 1
                currently_in_peak = False

        # NOT else, currently_in_peak may have changed!
        if not currently_in_peak:
            if current_peak < n_peaks and i == raw_peaks[current_peak, 0]:
                currently_in_peak = True
            else:
                delta = x - mean
                # Update n, mean and m2
                n += 1
                mean += delta/n
                m2 += delta*(x-mean)

    # Put results in result
    result[0] = mean
    if n < 2:
        result[1] = 0
    else:
        result[1] = (m2/n)**0.5


@numba.jit(numba.int64(numba.float64[:], numba.float64, numba.float64, numba.int64[:, :]), nopython=True)
def _numba_find_peaks(w, noise_sigma, threshold, intervals):
    """Fills intervals with left, right indices of intervals > noise_sigma which exceed threshold somewhere
     intervals: numpy () of [-1,-1] lists, will be filled by function.
    Returns: number of intervals found
    """

    in_candidate_interval = False
    current_interval_passed_test = False
    current_interval = 0

    for i, x in enumerate(w):

        if not in_candidate_interval and x > noise_sigma:
            # Start of candidate interval
            in_candidate_interval = True
            intervals[current_interval, 0] = i

        # This must be if, not else: an interval can cross threshold in start sample
        if in_candidate_interval:

            if x > threshold:
                current_interval_passed_test = True

            if x < noise_sigma:
                # End of candidate interval
                if current_interval_passed_test:
                    intervals[current_interval, 1] = i - 1
                    current_interval += 1
                in_candidate_interval = False
                current_interval_passed_test = False

    # Add last interval, if it didn't end
    # TODO: Hmm, should this raise a warning?
    if in_candidate_interval and current_interval_passed_test:
        intervals[current_interval, 1] = len(w) - 1
        current_interval += 1
    else:
        intervals[current_interval, 0] = 0

    return current_interval
