import numpy as np
from pax import plugin, datastructure, dsputils, units

# Used for diagnostic plotting only
# TODO: factor out to separate plugin?
import matplotlib.pyplot as plt
import os

import pandas


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

        self.channels_in_detector = {
            'tpc':  self.config['pmts_top'] | self.config['pmts_bottom'],
        }
        for det, chs in self.config['external_detectors'].items():
            self.channels_in_detector[det] = chs

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

        # Handle each detector separately
        for detector in self.channels_in_detector.keys():

            # Get all free regions before the give_up_after point
            for region_left, region_right in dsputils.free_regions(event, detector):

                # Can we give up yet?
                if region_left >= give_up_after:
                    break

                # Find all occurrences completely enveloped in the free region. Thank pyintervaltree for log(n) runtime
                # TODO: we should put strict=False, so we're not relying on the zero-suppression to separate
                # small peaks close to a large peak. However, right now this brings in stuff from large peaks if their
                # boundsare not completely tight...
                ocs = event.occurrences_interval_tree.search(region_left, region_right, strict=True)
                self.log.debug("Free region %05d-%05d: process %s occurrences" % (region_left, region_right, len(ocs)))

                for oc in ocs:
                    # Focus only on the part of the occurrence inside the free region (superfluous as long as strict=True)
                    # Remember: intervaltree uses half-open intervals, stop is the first index outside
                    start = max(region_left, oc.begin)
                    stop = min(region_right + 1, oc.end)
                    channel = oc.data['channel']

                    # Don't consider channels from other detectors
                    if channel not in self.channels_in_detector[detector]:
                        continue

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




class ClusterAndClassifySmallPeaks(plugin.TransformPlugin):

    def startup(self):
        self.dt = dt = self.config['digitizer_t_resolution']
        self.cluster_separation_length = self.config['cluster_separation_time']
        self.channels_in_detector = {
            'tpc':  self.config['pmts_top'] | self.config['pmts_bottom'],
        }
        for det, chs in self.config['external_detectors'].items():
            self.channels_in_detector[det] = chs

        ## Build the waveform models
        # self.log.info("Building waveform models...")
        # self.simulator = self.processor.simulator
        # self.wvmodels = {}
        # for rt in ('ER', 'NR'):
        #     # Infinite S1s
        #     self.wvmodels['s1_%s_inf'%rt] = WaveformModel(
        #         training_set=self.processor.simulator.s1_photons(100000, rt),
        #         zero_function=np.min,
        #         base_type='s1'
        #     )
        #
        #     # Small S1s: 3 pe for demonstration
        #     # mean normalization introduces a sample-size dependent effect
        #     self.wvmodels['s1_%s_3'%rt] = WaveformModel(
        #         training_set=self.simulator.s1_photons(3*10000, rt),
        #         zero_function=np.median,
        #         sample_length=3,
        #         base_type='s1'
        #     )
        #
        # # Deeper S2s
        # # TODO: depth=10 problematic. Can't just mix in a few depth=10, there's drift time...
        # self.wvmodels['s2_10cm'] = WaveformModel(
        #     training_set=self.simulator.s2_scintillation(self.simulator.s2_electrons(10000, z=10*units.cm)),
        #     zero_function=np.mean,
        #     base_type='s2'
        # )
        #
        # # Surface S2s (i.e. including single-electron S2s)
        # self.wvmodels['s2_surface'] = WaveformModel(
        #     training_set=self.simulator.s2_scintillation(electron_arrival_times=np.zeros(1000)),
        #     zero_function=np.mean,
        #     base_type='s2'
        # )
        # self.log.info("...done")
        #
        # import matplotlib.pyplot as plt
        #
        # def make_plot(func, xmin, xmax, dx, *args, **kwargs):
        #     x = np.linspace(xmin, xmax, (xmax-xmin)/dx)
        #     y = list(map(func, x))
        #     plt.plot(x, y, *args, **kwargs)
        #
        # for wv_name, wv_model in self.wvmodels.items():
        #     make_plot(wv_model.stat.pdf, -units.us, units.us, 10*units.ns, label=wv_name)
        # plt.yscale('log')
        # plt.legend()
        # plt.xlabel('time (ns) from zero')
        # plt.ylabel('pdf')
        # plt.show()


    def transform_event(self, event):

        # Handle each detector separately
        for detector in self.channels_in_detector.keys():

            # Hmzz, python has no do_while, so..
            redo_classification = True
            while redo_classification:
                redo_classification = False

                # Get all single-pe data in a list of dicts, sorted by index_of_maximum
                spes = sorted([
                    p.to_dict() for p in event.channel_peaks
                                if p.channel in self.channels_in_detector[detector]
                                and (not self.config['exclude_bad_channels'] or p.channel not in event.bad_channels)
                ], key=lambda x: x['index_of_maximum'])

                times = [s['index_of_maximum'] * self.dt for s in spes]
                assert(times == sorted(times))
                time_clusters = dsputils.split_by_separation(times, self.cluster_separation_length, return_indices=True)

                # Make a list of dicts of spe clusters (essentially a dataframe, but I want to do a for loop...)
                clusters = [{
                        'spes':         cluster_spes,
                        'n_spes':       len(cluster_spes),
                        # TODO: add extents of min/max peaks
                        'left':         spes[cluster_spes[0]]['left'],
                        'right':        spes[cluster_spes[-1]]['right'],
                        'type':         'unknown',
                    } for cluster_spes in time_clusters]

                dark_count = {}

                for c in clusters:
                    # Find how many channels show something (noise, bad)
                    # and how many are good & show photons

                    coincident_occurrences = event.occurrences_interval_tree.search(c['left'], c['right'], strict=False)
                    c['channels_with_something'] = set(self.channels_in_detector[detector]) & \
                                                   set([oc.data['channel'] for oc in coincident_occurrences])
                    c['channels_with_photons'] = set([spes[x]['channel'] for x in c['spes']])
                    c_times = [times[i] for i in c['spes']]
                    c['mad'] = dsputils.mad(c_times)

                    if len(c['channels_with_something']) > 2 * len(c['channels_with_photons']):
                        c['type'] = 'noise'

                    elif len(c['channels_with_photons']) == 1:
                        c['type'] = 'lone_pulse'
                        channel = spes[c['spes'][0]]['channel']
                        dark_count[channel] = dark_count.get(channel, 0) + 1

                    else:

                        # # Use the waveform models for classification
                        # # TODO handle pile-up in WaveformModel: pass all info, not just times
                        # classification_results = []
                        # for wv_name, wv_model in self.wvmodels.items():
                        #     classification_results.append({
                        #         'name':              wv_name,
                        #         'base_type':         wv_model.base_type,
                        #         'ks_p':              wv_model.timing_kstest(c_times)[1],
                        #         'total_likelihood':  wv_model.total_likelihood(c_times)
                        #     })
                        #
                        # classification_results.sort(key=lambda x: x['total_likelihood'], reverse=True)
                        #
                        # print(pandas.DataFrame(classification_results))
                        #
                        # #TODO: store in event class
                        # classification_options = [c for c in classification_results
                        #                           if c['total_likelihood'] > -6]# and c['ks_p'] > 10**-6]
                        #
                        # if len(classification_options) == 0:
                        #     c['type'] = 'unknown'
                        # else:
                        #     # TODO: classify as unknown if likelihood of other base type is similar
                        #     c['type'] = max(classification_options, key=lambda x : x['total_likelihood'])['base_type']

                        if c['mad'] < 10:
                            c['type'] = 's1'
                        else:
                            if c['n_spes'] < 5:
                                c['type'] = 'unknown'
                            else:
                                c['type'] = 's2'

                # Look for channels with abnormal dark rate
                for ch, dc in dark_count.items():
                    if dc > self.config['maximum_lone_pulses_per_channel']:
                        self.log.debug(
                            "Channel %s shows an abnormally high lone pulse rate (%s): its spe pulses will be excluded" % (
                                ch, dc))
                        event.bad_channels.append(ch)
                        if self.config['exclude_bad_channels']:
                            redo_classification = True

            # Classification is now done, so add the peaks to the datastructure
            # clusters was last set in the while redo_classification loop...
            for c in clusters:
                # We need an index_of_maximum and height, these we can only get from the sum waveform]
                # TODO: do we really want these in datastructure even for peaks reconstructed from spes?
                sum_wave = event.get_waveform(detector).samples[c['left'] : c['right'] + 1]
                max_idx = np.argmax(sum_wave)
                height = sum_wave[max_idx]
                event.peaks.append(datastructure.Peak({
                    'index_of_maximum':     max_idx + c['left'],
                    'height':               height,
                    'left':                 c['left'],
                    'right':                c['right'],
                    'area':                 sum([spes[x]['area'] for x in c['spes']]),
                    'contributing_pmts':    np.array(list(c['channels_with_photons']), dtype=np.uint16),
                    'area_per_pmt':         np.array([
                                                sum([spes[x]['area'] for x in c['spes'] if spes[x]['channel'] == ch])
                                                for ch in range(len(event.pmt_waveforms))]),
                    'type':                 c['type'],
                    'detector':             detector,
                    'mean_absolute_deviation': c['mad'],
                }))

        return event


##
# Stuff for waveform modelling
##
import scipy
from scipy import stats, interpolate


class WaveformModel:

    def __init__(self, training_set, zero_function, sample_length=None, fudge=10*units.ns, base_type='weird'):

        self.base_type = base_type
        self.zero_function = zero_function

        # Add some timing fudge
        training_set += np.random.normal(0, fudge, len(training_set))

        # Apply mean normalization
        if sample_length is not None:
            # Chop into parts of length sample_length, apply zero_function to each
            training_set = training_set.reshape((-1, sample_length))
            training_set -= zero_function(training_set, axis=1)[:,np.newaxis]
            training_set = training_set.reshape((-1))
        else:
            training_set -= zero_function(training_set)

        # Poor man's binning - is this good enough or bad?
        # Hmzz it broke stuff
        #training_set = np.around(training_set/conf['digitizer_t_resolution'])*conf['digitizer_t_resolution']

        self.stat = MCStatistic.from_mc(training_set) #, n_points_for_cdf=len(np.unique(training_set)))

    def timing_kstest(self, sample):
        sample -= self.zero_function(sample)
        return stats.kstest(sample, self.stat.cdf)

    def total_likelihood(self, sample):
        # Child classes can override this to include additional penalties for e.g. n_photons, hitpattern, ...
        # TODO: not quite though, sample is still a plain list...
        return self.timing_likelihood(sample)

    def timing_likelihood(self, sample):

        sample -= self.zero_function(sample)
        log_ls = np.log10(self.stat.pdf(sample))

        #A single outlier should not push everything to -inf
        log_ls = np.clip(log_ls, -50, 0)
        log_ls[np.isnan(log_ls)] = -50

        return  1/len(sample) * np.sum(log_ls)


#TODO: belongs in separate package
class MCStatistic(scipy.stats.rv_continuous):

    #init defines cdf by monte carlo, lets scipy take care of everything else
    @classmethod
    def from_mc(cls, mc_training_sample=None, n_points_for_cdf=None, **kwargs):
        if n_points_for_cdf is None:
            n_points_for_cdf = int(2*len(mc_training_sample)**(0.6))
        s = mc_training_sample
        s.sort()
        # TODO: Maybe first interpolate all points, then use http://scipy-central.org/item/53/1/adaptive-sampling-of-1d-functions
        # Poor man's adaptive sampling...
        # First divide into chunks to determine how much data varies in this range
        n_chunks = n_points_for_cdf**0.5
        presample_points = np.linspace(0, len(s)-1, int(n_chunks)).astype('int')
        points_to_sample = np.diff(s[presample_points])**0.8 # Sample density ~ (data variation)
        points_to_sample *= n_points_for_cdf/np.sum(points_to_sample)
        points_to_sample += 1
        points_to_sample = points_to_sample.astype('int')
        # print(n_points_for_cdf, presample_points, points_to_sample)
        # Now sample all the ranges
        sampling_points = []
        for i in range(len(points_to_sample)):
            sampling_points.append(np.linspace(
                presample_points[i],
                presample_points[i+1],
                points_to_sample[i],
                endpoint=False
            ).astype('int'))
        sampling_points = np.concatenate(sampling_points)
        cdf_training_points = np.array((s[sampling_points],sampling_points/len(s)))
        return cls.from_cdf_training_points(cdf_training_points, **kwargs)

    @classmethod
    def from_cdf_training_points(cls, cdf_training_points, **kwargs):
        self = cls(**kwargs)
        self.cdf_training_points = cdf_training_points
        self.training_limits = (cdf_training_points[0][0], cdf_training_points[0][-1])
        #Constuct empirical cdf
        self.cdf_itp = interpolate.interp1d(*self.cdf_training_points)
        return self

    @classmethod
    def from_file(cls, filename):
        cls.from_cdf_training_points(np.load(filename))

    def to_file(self, filename):
        np.save(filename, self.cdf_training_points)

    def _cdf(self, x):
        # Support for single values
        try:
            x[0]
        except TypeError:
            x = np.array([x])
        # Clipping = Extrapolate cdf constant outside training limits
        # too scared to clip in-place..
        x_clipped = np.zeros(len(x))
        np.clip(x, self.training_limits[0], self.training_limits[1], x_clipped)
        return self.cdf_itp(x_clipped)
