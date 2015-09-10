import numpy as np
import numba

from pax import plugin, datastructure, utils


class NaturalBreaksClustering(plugin.TransformPlugin):
    """Split peaks by a variation on the 'natural breaks' algorithm.
    Any gaps (distances between hits) in peaks larger than min_gap_size_for_break are tested by computing a
    'goodness of split' for splitting the cluster at that point.
    If it is larger than min_split_goodness(n_hits), the cluster is split at that gap and the declustering is
    called recursively for the newly minted clusters.

    The threshold function min_split_goodness(n_hits) has to be chosen so that S1s and S2s are not split up.
    This can be done by simulating them with pax's integrated waveform simulator.
    Keep in mind that this simulation has to be re-done with every significant change in a detector property
    that affects the signal shape (in particular, a change in electric fields).
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.min_gap_size_for_break = self.config['min_gap_size_for_break'] / self.dt
        self.max_n_gaps_to_test = self.config['max_n_gaps_to_test']
        self.min_split_goodness = eval(self.config['min_split_goodness'])

    def transform_event(self, event):
        new_peaks = []
        for peak in event.peaks:
            new_peaks += self.cluster(peak)
        event.peaks = new_peaks
        return event

    def cluster(self, peak):
        """Cluster hits in peak by variant on the natural breaks algorithm.
        Returns list of new peaks constructed from the peak (if no change, will be list with one element).
        Will set interior_split_goodness and birthing_split_goodness attributes
        """
        hits = peak.hits
        n_hits = len(hits)

        # Handle trivial cases
        if n_hits == 0:
            raise RuntimeError("Empty list passed to decluster!")
        elif n_hits == 1:
            # Lone hit: can't cluster any more!
            return [peak]

        left = np.zeros(len(hits))
        right = np.zeros(len(hits))
        area = np.zeros(len(hits))
        for i, h in enumerate(hits):
            left[i] = h.left
            area[i] = h.area
            right[i] = h.right
        area_tot = np.sum(area)
        gaps = utils.gaps_between_hits(hits)[1:]            # Remember first "gap" is zero: throw it away
        self.log.debug("Got %d hits (%d-%d) to decluster" % (len(hits), left[0], right[-1]))

        # Get indices of the self.max_n_gaps_to_test largest gaps
        split_threshold = self.min_split_goodness(n_hits)
        max_split_goodness = float('-inf')
        max_split_goodness_i = 0
        for gap_i in indices_of_largest_n(gaps, self.max_n_gaps_to_test):
            if gaps[gap_i] < self.min_gap_size_for_break:
                break

            # Index of hit to split on = index first hit that will go to right cluster
            split_i = gap_i + 1

            # Compute the naturalness of this break
            split_goodness = compute_split_goodness(split_i, left, right, area)

            # Should we split? If so, recurse.
            if split_goodness > self.min_split_goodness(n_hits):
                self.log.debug("SPLITTING at %d  (%s > %s)" % (split_i, split_goodness, split_threshold))
                peak_l = datastructure.Peak(hits=hits[:split_i],
                                            detector=peak.detector,
                                            birthing_split_goodness=split_goodness,
                                            birthing_split_fraction=np.sum(area[:split_i]) / area_tot)
                peak_r = datastructure.Peak(hits=hits[split_i:],
                                            detector=peak.detector,
                                            birthing_split_goodness=split_goodness,
                                            birthing_split_fraction=np.sum(area[split_i:]) / area_tot)
                return self.cluster(peak_l) + self.cluster(peak_r)
            else:
                self.log.debug("Proposed split at %d not good enough (%s < %s)" % (split_i,
                                                                                   split_goodness,
                                                                                   self.min_split_goodness(n_hits)))
            if split_goodness >= max_split_goodness:
                max_split_goodness = split_goodness
                max_split_goodness_i = split_i

        # If we get here, no clustering was needed
        peak.interior_split_goodness = max_split_goodness
        if max_split_goodness_i != 0:
            peak.interior_split_fraction = min(np.sum(area[:max_split_goodness_i]),
                                               np.sum(area[max_split_goodness_i:])) / area_tot
        return [peak]


@numba.jit(nopython=True)
def compute_split_goodness(split_index, left, right, area):
    """Return "goodness of split" for splitting hits >= split_index into right cluster, < into left.
       left, right: left, right indices of hits
       area: area of hits
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
    if split_index >= len(left) or split_index <= 0:
        raise ValueError("Ridiculous split index received!")
    numerator = _sad_two(left, right, weights=area)
    denominator = _sad_two(left[:split_index], right[:split_index], weights=area[:split_index])
    denominator += _sad_two(left[split_index:], right[split_index:], weights=area[split_index:])
    denominator += left[split_index] - right[split_index - 1]
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
