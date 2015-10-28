import numpy as np
import numba
from scipy.interpolate import InterpolatedUnivariateSpline

from pax import plugin, datastructure
from pax import dsputils


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
        self.min_split_goodness = InterpolatedUnivariateSpline(*self.config['split_goodness_threshold'])

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

        area_tot = np.sum(hits['area'])
        gaps = dsputils.gaps_between_hits(hits)[1:]            # Remember first "gap" is zero: throw it away
        self.log.debug("Clustering hits %d-%d" % (hits[0]['center'], hits[-1]['center']))

        # Get indices of the self.max_n_gaps_to_test largest gaps
        split_threshold = self.min_split_goodness(np.log10(area_tot))
        max_split_goodness = float('-inf')
        max_split_goodness_i = 0
        for gap_i in indices_of_largest_n(gaps, self.max_n_gaps_to_test):
            if gaps[gap_i] < self.min_gap_size_for_break:
                self.log.debug('Breaking because gap size %d smaller than %d' % (gaps[gap_i],
                                                                                 self.min_gap_size_for_break))
                break

            # Index of hit to split on = index first hit that will go to right cluster
            split_i = gap_i + 1

            # Compute the naturalness of this break
            split_goodness = compute_split_goodness(split_i,
                                                    hits['center'], hits['sum_absolute_deviation'], hits['area'])

            # Should we split? If so, recurse.
            if split_goodness > split_threshold:
                self.log.debug("SPLITTING at %d  (%s > %s)" % (split_i, split_goodness, split_threshold))
                peak_l = datastructure.Peak(hits=hits[:split_i],
                                            detector=peak.detector,
                                            birthing_split_goodness=split_goodness,
                                            birthing_split_fraction=np.sum(hits['area'][:split_i]) / area_tot)
                peak_r = datastructure.Peak(hits=hits[split_i:],
                                            detector=peak.detector,
                                            birthing_split_goodness=split_goodness,
                                            birthing_split_fraction=np.sum(hits['area'][split_i:]) / area_tot)
                return self.cluster(peak_l) + self.cluster(peak_r)
            else:
                self.log.debug("Proposed split at %d not good enough (%0.3f < %0.3f)" % (
                    split_i, split_goodness, split_threshold))
            if split_goodness >= max_split_goodness:
                max_split_goodness = split_goodness
                max_split_goodness_i = split_i

        # If we get here, no clustering was needed
        peak.interior_split_goodness = max_split_goodness
        if max_split_goodness_i != 0:
            peak.interior_split_fraction = min(np.sum(hits['area'][:max_split_goodness_i]),
                                               np.sum(hits['area'][max_split_goodness_i:])) / area_tot
        return [peak]


@numba.jit(numba.float64(numba.float64[:], numba.float64[:], numba.float64[:]),
           nopython=True, cache=True)
def _sad_fallback(x, areas, fallback):
    # While there is a one-pass algorithm for variance, I haven't found one for sad.. maybe it doesn't exists
    # First calculate the weighted mean.
    mean = 0
    sum_weights = 0
    for i in range(len(x)):
        mean += x[i] * areas[i]
        sum_weights += areas[i]
    mean /= sum_weights

    # Now calculate the sum abs dev, ensuring each x contributes at least fallback
    sad = 0
    for i in range(len(x)):
        sad += max(fallback[i], abs(x[i] - mean)) * areas[i]
    return sad


@numba.jit(numba.float64(numba.int64, numba.float64[:], numba.float64[:], numba.float64[:]),
           nopython=False, cache=True)
def compute_split_goodness(split_index, center, deviation, area):
    """Return "goodness of split" for splitting hits >= split_index into right cluster, < into left.
       left, right: left, right indices of hits
       area: area of hits
    "goodness of split" = 1 - (sad(left cluster) + sad(right cluster) / sad(all_hits)
      where sad = weighted (by area) sum of absolute deviation from mean.
    For more information, see this note:
    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:processor:natural_breaks_clustering
    """
    if split_index > len(center) - 1 or split_index <= 0:
        raise ValueError("Ridiculous split index received!")
    numerator = _sad_fallback(center[:split_index], areas=area[:split_index], fallback=deviation[:split_index])
    numerator += _sad_fallback(center[split_index:], areas=area[split_index:], fallback=deviation[split_index:])
    denominator = _sad_fallback(center, areas=area, fallback=deviation)
    return 1 - numerator / denominator


def indices_of_largest_n(a, n):
    """Return indices of n largest elements in a"""
    if len(a) == 0:
        return []
    return np.argsort(-a)[:min(n, len(a))]
