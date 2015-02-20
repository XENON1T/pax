import numpy as np
import numba

from pax import plugin, datastructure, units


# Used for diagnostic plotting only
# TODO: factor out to separate plugin?
import matplotlib.pyplot as plt
import os


class FindSmallPeaks(plugin.TransformPlugin):

    def startup(self):

        # Get settings from configuration
        self.min_sigma = self.config['peak_minimum_sigma']
        self.initial_noise_sigma = self.config['noise_sigma_guess']
        self.max_hits_per_occurrence = self.config['max_hits_per_occurrence']

        # Optional settings
        self.filter_to_use = self.config.get('filter_to_use', None)
        self.max_noise_detection_passes = self.config.get('max_noise_detection_passes', float('inf'))
        self.make_diagnostic_plots = self.config.get('make_diagnostic_plots', 'never')
        self.make_diagnostic_plots_in = self.config.get('make_diagnostic_plots_in', 'small_pf_diagnostic_plots')
        if self.make_diagnostic_plots != 'never':
            if not os.path.exists(self.make_diagnostic_plots_in):
                os.makedirs(self.make_diagnostic_plots_in)

    def transform_event(self, event):
        # ocs is shorthand for occurrences, as usual
        noise_count = {}

        for oc in event.occurrences:
            start = oc.left
            stop = oc.right
            channel = oc.channel

            # Don't consider dead channels
            if self.config['gains'][channel] == 0:
                continue

            # Retrieve the waveform from channel_waveforms
            w = event.channel_waveforms[channel, start: stop + 1]

            noise_sigma = self.initial_noise_sigma
            old_raw_peaks = []
            pass_number = 0
            baseline_correction = 0

            # Use several passes to separate noise / peaks
            for pass_number in range(self.max_noise_detection_passes):

                # Determine the peaks based on the noise level
                # Can't just use w > self.min_sigma * noise_sigma here,
                # want to extend peak bounds to noise_sigma
                raw_peaks = np.zeros((self.max_hits_per_occurrence, 2), dtype=np.int64)
                n_raw_peaks_found = self._find_peaks(w,
                                                     self.min_sigma * noise_sigma,
                                                     noise_sigma,
                                                     raw_peaks)
                if n_raw_peaks_found == -1:
                    # We found > self.max_hits_per_occurrence: write this in the debug log
                    self.log.debug("Occurrence %s-%s in channel %s has more than %s hits on peakfinding "
                                   "pass %s! Probably the noise level is unusually high. Further hits "
                                   "will be ignored." % (start, stop, channel,
                                                         self.max_hits_per_occurrence,
                                                         pass_number))
                    n_raw_peaks_found = len(raw_peaks)
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
                self._mean_std_outside_peaks(w, raw_peaks, result)
                baseline_correction_delta = result[0]
                noise_sigma = result[1]

                # Perform the baseline correction
                w -= baseline_correction_delta
                baseline_correction += baseline_correction_delta

                old_raw_peaks = raw_peaks

            else:
                self.log.warning((
                    "In occurrence %s-%s in channel %s, findSmallPeaks did not converge on peaks after %s" +
                    " iterations. This could indicate a baseline problem in this occurrence. " +
                    "Channel-based peakfinding in this occurrence may be less accurate.") % (
                        start, stop, channel, pass_number))

            # Update occurrence data
            oc.noise_sigma = noise_sigma
            oc.baseline_correction = baseline_correction
            oc.height = oc.height - baseline_correction

            # Update the noise occurrence count
            if len(raw_peaks) == 0:
                noise_count[channel] = noise_count.get(channel, 0) + 1

            # Compute peak area and max
            argmaxes = np.zeros(len(raw_peaks), dtype=np.int64)
            areas = np.zeros(len(raw_peaks))
            self._peak_argmax_and_area(w, raw_peaks, argmaxes, areas)

            # Store the found peaks in the datastructure
            peaks = []
            for i, p in enumerate(raw_peaks):
                # TODO: Hmzz.. height is computed in unfiltered waveform, noise_level in filtered!
                # TODO: height is wrong if baseline is corrected...
                # TODO: are you sure it is ok to compute area in filtered waveform?
                max_idx = p[0] + argmaxes[i]
                peaks.append(datastructure.ChannelPeak({
                    # TODO: store occurrence index
                    'channel':             channel,
                    'left':                start + p[0],
                    'index_of_maximum':    start + max_idx,
                    'right':               start + p[1],
                    'area':                areas[i],
                    'height':              w[max_idx],
                    'noise_sigma':         noise_sigma,
                }))
            event.all_channel_peaks.extend(peaks)

            # TODO: move to separate plugin?
            if self.make_diagnostic_plots == 'always' or \
               self.make_diagnostic_plots == 'no peaks' and not len(peaks):
                plt.figure()
                plt.plot(w, drawstyle='steps', label='data')
                for p in raw_peaks:
                    plt.axvspan(p[0] - 1, p[1], color='red', alpha=0.5)
                plt.plot(noise_sigma * np.ones(len(w)), '--', label='1 sigma')
                plt.plot(self.min_sigma * noise_sigma * np.ones(len(w)),
                         '--', label='%s sigma' % self.min_sigma)
                plt.legend()
                bla = (event.event_number, start, stop, channel)
                plt.title('Event %s, occurrence %d-%d, Channel %d' % bla)
                plt.xlabel("Sample [%s ns]" % (event.sample_duration / units.ns))
                plt.ylabel("Amplitude [pe]")
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

    # TODO: Needs a test!
    @staticmethod
    @numba.jit(numba.void(numba.float64[:], numba.int64[:, :], numba.int64[:], numba.float64[:]), nopython=True)
    def _peak_argmax_and_area(w, raw_peaks, argmaxes, areas):
        for peak_i in range(len(raw_peaks)):
            current_max = -999.9
            current_argmax = -1
            current_area = 0
            for i, x in enumerate(w[raw_peaks[peak_i, 0]:raw_peaks[peak_i, 1]+1]):
                if x > current_max:
                    current_max = x
                    current_argmax = i
                current_area += x
            argmaxes[peak_i] = current_argmax
            areas[peak_i] = current_area

    @staticmethod
    @numba.jit(numba.void(numba.float64[:], numba.int64[:, :], numba.float64[:]), nopython=True)
    def _mean_std_outside_peaks(w, raw_peaks, result):
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

    @staticmethod
    @numba.jit(numba.int64(numba.float64[:], numba.float64, numba.float64, numba.int64[:, :]), nopython=True)
    def _find_peaks(w, threshold, bound_threshold, intervals):
        """Fills intervals with left, right indices of intervals > bound_threshold which exceed threshold somewhere
         intervals: numpy () of [-1,-1] lists, will be filled by function.
        Returns: number of intervals found
        Will stop search after intervals found reached length of intervals argument passed in
        """

        in_candidate_interval = False
        current_interval_passed_test = False
        current_interval = 0
        max_intervals = len(intervals) - 1

        for i, x in enumerate(w):

            if not in_candidate_interval and x > bound_threshold:
                # Start of candidate interval
                in_candidate_interval = True
                intervals[current_interval, 0] = i

            # This must be if, not else: an interval can cross threshold in start sample
            if in_candidate_interval:

                if x > threshold:
                    current_interval_passed_test = True

                if x < bound_threshold:
                    # End of candidate interval
                    if current_interval_passed_test:
                        # We've found a new peak!
                        intervals[current_interval, 1] = i - 1
                        current_interval += 1
                        if current_interval > max_intervals:
                            # We found more peaks than we have room in our result array!
                            # Would love to raise an exception, but can't...
                            # Instead just return -1, caller will have to check this
                            return -1
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
