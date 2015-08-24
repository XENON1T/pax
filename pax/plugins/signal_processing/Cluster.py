import numpy as np
import numba
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


class NaturalBreaks(ClusterPlugin):
    """Split hits by a variation on the 'natural breaks' algorithm.
    The clustering proceeds in two steps:
     1) Hits are clustered by breaking whenever a gap larger than max_gap_size_in_cluster is encountered.
     2) Next, These clusters are declustered. Any gaps larger than min_gap_size_for_break are tested by computing
        a 'goodness of split' for splitting the cluster at that point.
        If it is larger than min_split_goodness(n_hits), the cluster is split at that gap and the declustering is
         called recursively for the newly minted clusters.

    The threshold function min_split_goodness(n_hits) has to be chosen so that S1s and S2s are not split up.
    This can be done by simulating them with pax's integrated waveform simulator.
    Keep in mind that this simulation has to be re-done with every significant change in a detector property
    that affects the signal shape (in particular, a change in electric fields).
    """

    def startup(self):
        super().startup()
        self.max_gap_size_in_cluster = self.config['max_gap_size_in_cluster'] / self.dt
        self.min_gap_size_for_break = self.config['min_gap_size_for_break'] / self.dt
        self.max_n_gaps_to_test = self.config['max_n_gaps_to_test']
        self.min_split_goodness = eval(self.config['min_split_goodness'])

    def cluster_hits(self, hits):
        # Store basic quantities as attributes for access in several methods
        # Hits are already sorted by left
        self.left = np.array([h.left for h in hits])
        self.right = np.array([h.right for h in hits])
        self.area = np.array([h.area for h in hits])
        self.gaps = np.zeros_like(self.left)

        if len(hits) == 0:
            return []

        # Cluster hits by breaking on large enough gaps
        # gap = distance between hit left and largest right of any hits before it (lower left)
        # The first hit is always easy:
        farthest_right = hits[0].right
        clusters = []
        current_hits = [0]

        # First hit has no defined gap size: it won't be taken into account in the median
        self.gaps[0] = 0

        for i, hit in enumerate(hits):
            if i == 0:
                # We already dealt with the first hit
                continue

            if hit.left > farthest_right + self.max_gap_size_in_cluster:
                # This hit belongs to a new cluster. Declustering the cluster we just finished.
                clusters += self.decluster(current_hits)
                current_hits = []
                self.gaps[i] = 0

            else:
                # If the gap size is negative, the hit is totally inside a larger hit
                # We'll record a gap size of 0 instead
                self.gaps[i] = max(0, hit.left - farthest_right)

            current_hits.append(i)

            # Update the farthest right value seen in this cluster
            farthest_right = max(farthest_right, hit.right)

        # Decluster the final cluster
        clusters += self.decluster(current_hits)

        return clusters

    def decluster(self, hit_indices):
        """Decluster hits by variant on natural breaks algorithm. Return list of lists of hit_indices."""
        # Handle trivial cases
        if len(hit_indices) == 0:
            raise RuntimeError("Empty list passed to decluster!")
        elif len(hit_indices) == 1:
            return [hit_indices]

        self.log.debug("Got %d hits (%d-%d) to decluster" % (len(hit_indices),
                                                             self.left[hit_indices[0]],
                                                             self.right[hit_indices[-1]]))

        # Which gaps should we test?
        # The first gap in this cluster is not a meaningful quantity
        # (it's the distance to the last hit from the previous cluster)
        gaps = self.gaps[hit_indices[1:]]

        # Get indices of the self.max_n_gaps_to_test largest gaps
        for gap_i in indices_of_largest_n(gaps, self.max_n_gaps_to_test):
            if gaps[gap_i] < self.min_gap_size_for_break:
                break

            # Index of hit index in hit_indices to split hits on :-)
            split_i = gap_i + 1

            # Compute the natural break quantity for this break
            split_goodness = self.split_goodness(hit_indices, split_i)

            # Should we split? If so, call decluster recursively
            n = len(hit_indices)
            if split_goodness > self.min_split_goodness(n):
                self.log.debug("SPLITTING at %d  (%s > %s)" % (split_i,
                                                               split_goodness,
                                                               self.min_split_goodness(n)))
                return self.decluster(hit_indices[:split_i]) + self.decluster(hit_indices[split_i:])
            else:
                self.log.debug("Proposed split at %d not good enough (%s < %s)" % (split_i,
                                                                                   split_goodness,
                                                                                   self.min_split_goodness(n)))

        # If we get here, no declustering needed
        return [hit_indices]

    def split_goodness(self, hit_indices, split_index):
        """Return "goodness of split" for splitting everything >= split_index into right cluster, < into left.
        "goodness of split" = 0.5 * sad(all_hits) / (sad(left cluster) + sad(right cluster) + gap between clusters) - 1
          where sad = weighted (by area) sum of absolute deviation from mean,
          calculated on all *endpoints* of hits in the cluster
          The gap between clusters is added in the denominator as a penalty term against splitting very small signals.
          The reason we use sum absolute deviation instead of the root of the sum square deviation is that the latter
           is more sensitive to outliers. Clustering should NOT trim tails of peaks.
        This usually takes a value around [-1, 1], but can go much higher if the split is good.

        The reason we use endpoints, rather than hit centers, is compatibility with high-energy signals.
        These can have a single long hit in all channels, which won't have any short hits near to its center.
        """
        # If the performance of this function becomes a bottleneck, we can replace it with a numba algorithm.
        hit_indices = np.asarray(hit_indices)
        if split_index >= len(hit_indices) or split_index <= 0:
            raise ValueError("%d is a ridiculous split index for %d hits" % len(hit_indices), split_index)
        numerator = _sad_two(self.left[hit_indices], self.right[hit_indices], weights=self.area[hit_indices])
        denominator = _sad_two(self.left[hit_indices[:split_index]],
                               self.right[hit_indices[:split_index]],
                               weights=self.area[hit_indices[:split_index]],)
        denominator += _sad_two(self.left[hit_indices[split_index:]],
                                self.right[hit_indices[split_index:]],
                                weights=self.area[hit_indices[split_index:]],)
        denominator += self.left[hit_indices[split_index]] - self.right[hit_indices[split_index - 1]]
        return -1 + 0.5 * numerator / denominator


@numba.jit(nopython=True)
def _sad_two(x1, x2, weights):
    """Returns the weighted sum absolute deviation from the weighted mean of x1 and x2 (considered as one array)
    x1 and x2 must have same length.
    """
    # First calculate the weighted mean.
    # While there is a one-pass algorithm for variance, I haven't found one for sad.. maybe it doesn't exists
    mean = 0
    sum_weights = 0
    for i in range(len(x1)):
        mean += x1[i] * weights[i]
        mean += x2[i] * weights[i]
        sum_weights += 2 * weights[i]
    mean /= sum_weights

    # Now calculate the weighted sum absolute deviation
    sad = 0
    for i in range(len(x1)):
        sad += abs(x1[i] - mean) * weights[i]
        sad += abs(x2[i] - mean) * weights[i]
    # To normalize to the case with all weights 1, we multipy by n / sum_w
    sad *= 2 * len(x1) / sum_weights

    return sad


def indices_of_largest_n(a, n):
    """Return indices of n largest elements in a"""
    if len(a) == 0:
        return []
    return np.argsort(-a)[:min(n, len(a))]
