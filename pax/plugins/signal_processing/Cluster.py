import numpy as np
from scipy.optimize import brenth
import sklearn.cluster
from pax import plugin, datastructure, utils


class ClusterPlugin(plugin.TransformPlugin):
    """Base plugin for clustering

    Individual channel peaks into groups, and labels them as noise / lone_pulse / unknown
    'unknown' means an S1 or S2, which will be decided by a later plugin
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.n_channels = self.config['n_channels']

    def cluster_hits(self, hits):
        raise NotImplementedError

    def transform_event(self, event):

        # Handle each detector separately
        for detector in self.config['channels_in_detector'].keys():
            self.log.debug("Clustering channel peaks in data from %s" % detector)
            peaks = []      # Superfluous, while loop is always run once... but pycharm complains if we omit

            # Hmzz, python has no do_while, so..
            redo_classification = True
            while redo_classification:
                redo_classification = False
                peaks = []
                dark_count = {}

                # Get all single-pe data in a list of dicts, sorted by index_of_maximum
                spes = sorted([p for p in event.all_channel_peaks if p.channel in self.config['channels_in_detector'][detector] and (not self.config['exclude_bad_channels'] or not event.is_channel_bad[p.channel])],  # noqa, we're replacing this soon anyways
                              key=lambda s: s.index_of_maximum)
                self.log.debug("Found %s channel peaks" % len(spes))

                if not spes:
                    break

                ##
                # CALL THE CHILD CLASS' cluster_hits method
                ##

                # Cluster the single-pes in groups separated by >= self.s2_width
                cluster_indices = self.cluster_hits(spes)
                self.log.debug("Made %s clusters" % len(cluster_indices))

                # Each cluster becomes a peak
                # Compute basic properties, check for too many lone pulses per channel
                for ci in cluster_indices:
                    peak = datastructure.Peak({
                        'channel_peaks':            [spes[cidx] for cidx in ci],
                        'detector':                 detector,
                        'area_per_channel':         np.zeros(self.n_channels),
                        'does_channel_contribute':  np.zeros(self.n_channels, dtype='bool'),
                        'does_channel_have_noise':  np.zeros(self.n_channels, dtype='bool'),
                    })

                    # Compute contributing and noise channels - needed for bad channel rejection

                    # Contributing channels are in the detector and have an spe
                    # (not in a bad channel, but those spes have already been filtered out)
                    for s in peak.channel_peaks:
                        peak.does_channel_contribute[s.channel] = True

                    if peak.number_of_contributing_channels == 0:
                        raise RuntimeError(
                            "Every peak should have at least one contributing channel... what's going on?")

                    # Find how many channels show some data, but no spe
                    coincident_occurrences = event.get_occurrences_between(peak.left, peak.right, strict=False)
                    for oc in coincident_occurrences:
                        ch = oc.channel
                        if not peak.does_channel_contribute[ch]:
                            peak.does_channel_have_noise[ch] = True

                    # Classification for noise and lone_pulse peaks
                    if peak.number_of_noise_channels > 2 * peak.number_of_contributing_channels:
                        peak.type = 'noise'

                    elif peak.number_of_contributing_channels == 1:
                        peak.type = 'lone_pulse'
                        channel = peak.channel_peaks[0].channel
                        dark_count[channel] = dark_count.get(channel, 0) + 1

                    else:
                        # Proper peak, classification done later
                        peak.type = 'unknown'

                    peaks.append(peak)

                # Look for channels with abnormal dark rate
                for ch, dc in dark_count.items():
                    if dc > self.config['maximum_lone_pulses_per_channel']:
                        self.log.debug(
                            "Channel %s shows an abnormally high lone pulse rate (%s): marked as bad." % (ch, dc))
                        event.is_channel_bad[ch] = True
                        if self.config['exclude_bad_channels']:
                            redo_classification = True
                            self.log.debug("Clustering has to be redone!!")

            # Add the peaks to the datastructure
            event.peaks.extend(peaks)

        return event


class MeanShift(ClusterPlugin):
    """Clusters hits using mean-shift algorithm

    http://en.wikipedia.org/wiki/Mean_shift
    """

    def startup(self):
        super().startup()

        self.s2_size = self.config['s2_size']
        self.s2_width = self.config['s2_width']
        self.p_value = self.config['p_value']
        self.cluster_all = self.config['cluster_all']

        self.bandwidth = self.get_gap_size()
        self.log.info("Using bandwidth of %0.2f ns" % self.bandwidth)

    def cluster_hits(self, spes):
        # Cluster the single-pes in groups separated by >= self.s2_width
        cluster_indices = utils.cluster_by_diff([s.index_of_maximum * self.dt for s in spes],
                                                self.s2_width,
                                                return_indices=True)
        self.log.debug("Pre-clustering made %s clusters" % len(cluster_indices))

        # Make spes a numpy array, so we can do list-based indexing
        spes = np.array(spes)

        new_cluster_indices = []
        for ci in cluster_indices:
            ci = np.array(ci)
            channel_peak_objects = spes[[ci]]

            locations = np.arange(channel_peak_objects.size)
            areas = [int(s.area + 0.5) for s in channel_peak_objects]

            times_repeated = []
            for s in channel_peak_objects:
                for value in np.linspace(s.left, s.right, int(s.area + 0.5)):
                    times_repeated.append(value * self.dt)
            times_repeated = np.array(times_repeated, dtype=int)
            locations_repeated = np.repeat(locations, areas)

            data = np.vstack((times_repeated,
                              locations_repeated))
            if data.size == 0 or data[0].size == 0:
                continue

            x = data[0]
            X = np.array(list(zip(x, np.zeros(len(x)))),
                         dtype=np.int)

            ms = sklearn.cluster.MeanShift(bandwidth=self.bandwidth,
                                           bin_seeding=True,
                                           cluster_all=self.cluster_all)

            ms.fit(X)

            for label in np.unique(ms.labels_):
                if label == -1:  # Means no peak
                    continue
                peaks_in_cluster = np.unique(data[1,
                                                  (ms.labels_ == label)])
                new_cluster_indices.append(ci[peaks_in_cluster])

        return new_cluster_indices

    @staticmethod
    def get_gap_probability(gap_width, s2_size,
                            s2_width, p_value_offset=0.0,
                            n=10000):
        answer = 0.0

        for index in range(n):
            this_mu = np.random.poisson(s2_size)

            rand_numbers = np.random.random(size=this_mu)
            rand_numbers *= s2_width
            rand_numbers = np.append(rand_numbers, [0.0, s2_width])
            rand_numbers.sort()

            gap_sizes = rand_numbers[1:] - rand_numbers[0:-1]

            if np.max(gap_sizes) < gap_width:
                answer += 1
        return answer / n - p_value_offset

    def get_gap_size(self):
        return brenth(self.get_gap_probability,
                      0.0,
                      self.s2_width,
                      args=(self.s2_size,
                            self.s2_width,
                            self.p_value))


class HitDifference(ClusterPlugin):
    """Clusters hits based on times between their maxima
    If any hit maximum is separated by more than max_difference from the next,
    it starts a new cluster.
    """

    def cluster_hits(self, hits):
        return utils.cluster_by_diff([s.index_of_maximum * self.dt for s in hits],
                                     self.config['max_difference'],
                                     return_indices=True)


class HitGap(ClusterPlugin):
    """Clusters hits based on gaps = times not covered by any hits.
    Any gap longer than max_gap_size starts a new cluster.
    Difference with HitDifference: this interval nature of hits into account
    """

    def cluster_hits(self, hits):
        raise NotImplementedError
