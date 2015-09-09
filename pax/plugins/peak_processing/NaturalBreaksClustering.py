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
        peaks_to_delete = []
        for peak_i, peak in enumerate(event.peaks):
            hits = peak.hits
            # Store basic quantities as attributes for access in several methods
            # Hits are already sorted by left
            self.left = np.array([h.left for h in hits])
            self.right = np.array([h.right for h in hits])
            self.area = np.array([h.area for h in hits])
            self.gaps = utils.gaps_between_hits(hits)

            clusters = self.decluster(np.arange(len(hits)))

            if len(clusters) == 1:
                continue

            # Mark the current peak for deletion
            # We can't delete it now since we are iterating over peaks
            peaks_to_delete.append(peak_i)

            # Add new peaks for the newly found clusters
            for hit_indices in clusters:
                event.peaks.append(datastructure.Peak(detector=peak.detector))
                for hit_i in hit_indices:
                    event.peaks[-1].hits.append(hits[hit_i])

        # Remove the peaks marked for deletion
        event.peaks = [p for i, p in enumerate(event.peaks) if i not in peaks_to_delete]

        return event

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
