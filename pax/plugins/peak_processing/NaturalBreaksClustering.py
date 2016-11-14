import numpy as np
import numba
from scipy.interpolate import InterpolatedUnivariateSpline

from pax import plugin, datastructure
from pax import dsputils


class NaturalBreaksClustering(plugin.ClusteringPlugin):
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
        self.min_split_goodness = InterpolatedUnivariateSpline(*self.config['split_goodness_threshold'], k=1)
        self.dt = self.config['sample_duration']

    def cluster_peak(self, peak):
        """Cluster hits in peak by variant on the natural breaks algorithm.
        Returns list of new peaks constructed from the peak (if no change, will be list with one element).
        Will set interior_split_goodness and birthing_split_goodness attributes
        """
        hits = peak.hits
        hits.sort(order='left')
        n_hits = len(hits)

        # Handle trivial cases
        if n_hits == 0:
            raise RuntimeError("Empty list passed to decluster!")
        elif n_hits == 1:
            # Lone hit: can't cluster any more!
            return [peak]

        self.log.debug("Clustering hits %d-%d" % (hits[0]['center'], hits[-1]['center']))
        area_tot = np.sum(hits['area'])

        # Compute gaps between hits, select large enough gaps to test
        gaps = dsputils.gaps_between_hits(hits)[1:]            # Remember first "gap" is zero: throw it away
        selection = gaps > self.config['min_gap_size_for_break'] / self.dt
        split_indices = np.arange(1, len(gaps) + 1)[selection]
        gaps = gaps[selection]

        # Look for good split points
        gos_observed = np.zeros(len(gaps))
        compute_every_split_goodness(gaps, split_indices,
                                     hits['center'], hits['sum_absolute_deviation'], hits['area'],
                                     gos_observed)

        # Find the split point with the largest goodness of split
        if len(gos_observed):
            max_split_ii = np.argmax(gos_observed)
            split_i = split_indices[max_split_ii]
            split_goodness = gos_observed[max_split_ii]
            split_threshold = self.min_split_goodness(np.log10(area_tot))

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
                return self.cluster_peak(peak_l) + self.cluster_peak(peak_r)
            else:
                self.log.debug("Proposed split at %d not good enough (%0.3f < %0.3f)" % (
                    split_i, split_goodness, split_threshold))

            # If we get here, no clustering was needed
            peak.interior_split_goodness = split_goodness
            peak.interior_split_fraction = min(np.sum(hits['area'][:max_split_ii]),
                                               np.sum(hits['area'][max_split_ii:])) / area_tot

        return [peak]


@numba.jit(numba.float64(numba.float64[:], numba.float64[:], numba.float64[:]),
           nopython=True)
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


@numba.jit(numba.float64(numba.int64[:], numba.int64[:],
                         numba.float64[:], numba.float64[:], numba.float64[:],
                         numba.float64[:]),
           nopython=False)
def compute_every_split_goodness(gaps, split_indices,
                                 center, deviation, area,
                                 results):
    """Computes the "goodness of split" for several split points: see compute_split_goodness"""
    for gap_i, gap in enumerate(gaps):
        # Index of hit to split on = index first hit that will go to right cluster
        split_i = split_indices[gap_i]
        results[gap_i] = compute_split_goodness(split_i, center, deviation, area)


@numba.jit(numba.float64(numba.int64, numba.float64[:], numba.float64[:], numba.float64[:]),
           nopython=False)
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
