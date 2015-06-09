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

        # Build the channel -> detector lookup dict
        self.detector_by_channel = {}
        for name, chs in self.config['channels_in_detector'].items():
            for ch in chs:
                self.detector_by_channel[ch] = name

    def cluster_hits(self, hits):
        raise NotImplementedError

    def transform_event(self, event):

        # Sort hits by detector
        detectors = self.config['channels_in_detector'].keys()
        hits_per_detector = {detector: [] for detector in detectors}
        for hit in event.all_hits:
            hits_per_detector[self.detector_by_channel[hit.channel]].append(hit)

        # Handle each detector separately
        for detector, hits_to_cluster in hits_per_detector.items():

            # Sort hits by left index
            hits_to_cluster.sort(key=lambda x: x.left)

            # Grab pulses in this detector
            pulses = [oc for oc in event.pulses if oc.channel in self.config['channels_in_detector'][detector]]

            self.log.debug("Clustering channel peaks in data from %s" % detector)

            clustering_pass = 0
            while True:

                peaks = []

                # Do we actually have something to cluster?
                self.log.debug("Found %s channel peaks" % len(hits_to_cluster))
                if not hits_to_cluster:
                    break

                # Reset penalty
                penalty_per_ch = event.noise_pulses_in * self.config['penalty_per_noise_pulse']

                # Cluster the single-pes in groups /clusters which will become peaks
                # CHILD CLASS CONTAINING CLUSTERING ALGORITHM IS CALLED HERE
                indices_of_hits_per_peak = self.cluster_hits(hits_to_cluster)
                self.log.debug("Made %s clusters" % len(indices_of_hits_per_peak))

                # Each cluster becomes a peak
                # Compute basic properties, check for too many lone pulses per channel
                for this_peak_s_hit_indices in indices_of_hits_per_peak:

                    if len(this_peak_s_hit_indices) == 0:
                        raise RuntimeError("Every peak should have a hit... what's going on?")

                    peak = datastructure.Peak({
                        'hits':            [hits_to_cluster[i] for i in this_peak_s_hit_indices],
                        'detector':                 detector,
                        'area_per_channel':         np.zeros(self.n_channels),
                        'does_channel_contribute':  np.zeros(self.n_channels, dtype='bool'),
                        'does_channel_have_noise':  np.zeros(self.n_channels, dtype='bool'),
                    })

                    # Compute basic properties of peak, needed later in the plugin
                    # For speed it would be better to compute as much as possible later..
                    peak.left = peak.hits[0].left
                    peak.right = peak.hits[0].right
                    for hit in peak.hits:
                        peak.left = min(peak.left, hit.left)
                        peak.right = max(peak.right, hit.right)
                        peak.does_channel_contribute[hit.channel] = True
                    peak.n_contributing_channels = len(peak.contributing_channels)

                    if peak.n_contributing_channels == 0:
                        raise RuntimeError(
                            "Every peak should have at least one contributing channel... what's going on?")

                    # Find how many channels in this detector show some data at the same time as this peak, but no hit
                    # Maybe no hit was found, maybe hit was rejected by the suspicious channel algorithm
                    # If zero-length encoding is not used, all channels will have "noise" here
                    coincident_pulses = [pulse for pulse in pulses
                                         if pulse.left <= peak.right and pulse.right >= peak.left]
                    for oc in coincident_pulses:
                        channel = oc.channel
                        if not peak.does_channel_contribute[channel]:
                            peak.does_channel_have_noise[channel] = True
                    peak.n_noise_channels = len(peak.noise_channels)

                    # Classify noise and lone hits
                    # TODO: Noise classification should be configurable!
                    is_noise = peak.n_noise_channels / peak.n_contributing_channels > \
                        self.config['max_noise_channels_over_contributing_channels']
                    is_lone_hit = peak.n_contributing_channels == 1
                    if is_noise:
                        peak.type = 'noise'
                        # TODO: Should we also reject all hits?
                    elif is_lone_hit:
                        peak.type = 'lone_hit'
                        # Don't reject the hit(s) in this peak
                        # However, lone hits in suspicious channels will always be rejected later:
                        # in the suspicious channel algorithm since their 'witness area' is 0
                    else:
                        # Proper peak, classification done later
                        peak.type = 'unknown'

                    # Add a penalty for each hit, if the peak was lone / noise
                    if is_noise or is_lone_hit:
                        for hit in peak.hits:
                            penalty_per_ch[hit.channel] += self.config['penalty_per_lone_hit']

                    peaks.append(peak)

                # Are there any suspicious channels? If not, we are done.
                self.log.debug(', '.join(['%s: %s' % (ch, round(penalty_per_ch[ch], 1))
                                          for ch in np.where(penalty_per_ch > 0)[0]]))
                suspicious_channels = np.where(penalty_per_ch >=
                                               self.config['penalty_geq_this_is_suspicious'])[0]
                event.is_channel_suspicious[suspicious_channels] = True
                if len(suspicious_channels) == 0:
                    break

                # Compute good/bad area balance for each hit in a suspicious channel
                # Good area: area in non-suspicious channels in same peak
                # Bad area: area in lone pulses and noise in same channel + something extra for # of noise pulses
                rejected_some_hits = False
                for peak in peaks:

                    # Compute area for this peak outside suspicious channels
                    # a witness to this peak not being noise
                    witness_area = 0
                    for hit in peak.hits:
                        if hit.channel not in suspicious_channels:
                            witness_area += hit.area
                    # witness_area = peak.area - np.sum(peak.area_per_channel[suspicious_channels])

                    # If the witness area for a hit is lower than the penalty in that channel, reject the hit
                    for hit in peak.hits:
                        if hit.channel not in suspicious_channels:
                            continue
                        channel = hit.channel
                        if witness_area == 0 or penalty_per_ch[hit.channel] > witness_area:
                            hit.is_rejected = True
                            rejected_some_hits = True
                            event.n_hits_rejected[channel] += 1

                # If no new hits were rejected, we are done
                if not rejected_some_hits:
                    break

                # Delete rejected hits from hits_to_cluster, then redo clustering
                # The rejected hits will remain in event.all_hits, of course
                hits_to_cluster = [h for h in hits_to_cluster if not h.is_rejected]
                clustering_pass += 1

            # Add the peaks to the datastructure
            event.peaks.extend(peaks)
            self.log.debug("Clustering & bad channel rejection ended after %s passes" % (clustering_pass + 1))

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
        cluster_indices = utils.cluster_by_diff([s.center for s in spes],
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
        return utils.cluster_by_diff([s.center for s in hits],
                                     self.config['max_difference'],
                                     return_indices=True)


class GapSize(ClusterPlugin):

    """Clusters hits based on gaps = times not covered by any hits.
    Any gap longer than max_gap_size starts a new cluster.
    Difference with HitDifference: this takes interval nature of hits into account
    """

    def startup(self):
        super().startup()
        # Convert gap threshold to samples (is in time (ns) in config)
        self.large_gap_threshold = self.config['large_gap_threshold'] / self.dt
        self.small_gap_threshold = self.config['small_gap_threshold'] / self.dt
        self.transition_point = self.config['transition_point']

    def cluster_hits(self, hits):
        # If a hit has a left > this, it will form a new cluster
        boundary = -999999999
        clusters = []
        for i, hit in enumerate(hits):

            if hit.left > boundary:
                # Hit starts after current boundary: new cluster
                clusters.append([])
                # (Re)set area and thresholds
                area = 0
                gap_size_threshold = self.large_gap_threshold
                boundary = hit.right + gap_size_threshold

            # Add this hit to the cluster
            clusters[-1].append(i)
            area += hit.area

            # Can we start applying the tighter threshold?
            if area > self.transition_point:
                # Yes, there's no chance this is a single electron: use a tighter clustering boundary
                gap_size_threshold = self.small_gap_threshold

            # Extend the boundary at which a new clusters starts, if needed
            boundary = max(boundary, hit.right + gap_size_threshold)
        return clusters
