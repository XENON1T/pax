import numpy as np
import numba
from scipy.interpolate import InterpolatedUnivariateSpline

from pax import plugin, dsputils


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
        self.rescale_factor = self.config['rescale_factor']

    def cluster_peak(self, peak):
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

        self.log.debug("Clustering hits %d-%d" % (hits[0]['center'], hits[-1]['center']))

        # import matplotlib.pyplot as plt
        # for h in peak.hits:
        #     #plt.axvspan(h['left_central'], h['right_central'], color='k', alpha=0.1)
        #     if h['left_central'] < 515 and h['right_central'] > 515:
        #         print(h['channel'], h['found_in_pulse'])
        # plt.show()

        peak.hits.sort(order='left_central')
        selection = np.diff(peak.hits['left_central']) > 0
        split_indices = np.arange(1, len(peak.hits))[selection]

        # Look for good split points
        gos_observed = np.zeros(len(split_indices))
        compute_every_split_goodness(split_indices,
                                     hits['center'], hits['sum_absolute_deviation'], hits['area'],
                                     gos_observed, self.config['rescale_factor'])

        # Find the split point with the largest goodness of split
        if len(gos_observed):
            max_split_ii = np.argmax(gos_observed)
            split_i = split_indices[max_split_ii]
            split_goodness = gos_observed[max_split_ii]
            split_threshold = self.min_split_goodness(np.log10(peak.area))

            # Should we split? If so, recurse.
            if split_goodness > split_threshold:
                self.log.debug("SPLITTING at %d  (%s > %s)" % (split_i, split_goodness, split_threshold))
                # split_peak takes an index in the peak, not the event!
                peak_l, peak_r = self.split_peak(peak, [peak.hits[split_i]['left_central'] - peak.left - 1])
                return self.cluster_peak(peak_l) + self.cluster_peak(peak_r)
            else:
                self.log.debug("Proposed split at %d not good enough (%0.3f < %0.3f)" % (
                    split_i, split_goodness, split_threshold))

            # If we get here, no clustering was needed
            peak.interior_split_goodness = split_goodness
            peak.interior_split_fraction = min(np.sum(hits['area'][:max_split_ii]),
                                               np.sum(hits['area'][max_split_ii:])) / peak.area

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


@numba.jit(numba.float64(numba.int64, numba.float64[:], numba.float64[:], numba.float64[:], numba.float64),
           nopython=True)
def compute_split_goodness(split_index, center, deviation, area, rescale_factor):
    """Return "goodness of split" for splitting hits >= split_index into right cluster, < into left.
    High values indicate a better split.
       left, right: left, right indices of hits
       area: area of hits

    "goodness of split" = 1 - (f(left cluster) + f(right cluster) / f(all_hits)
    with f = mean absolute deviation from mean / area**rescale_factor
    So if rescale_factor = 1, f is the mean absolute deviation from the mean (weighted by area)
                         = 0, f is the SUM absolute deviation " '""
    Low rescale factors prefer symmetric (equal area) splits over non-symmetric ones.

    For more information, see this note:
    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:processor:natural_breaks_clustering
    """
    if split_index > len(center) - 1 or split_index <= 0:
        raise ValueError("Ridiculous split index received!")
    sad_right = _sad_fallback(center[:split_index], areas=area[:split_index], fallback=deviation[:split_index])
    sad_left = _sad_fallback(center[split_index:], areas=area[split_index:], fallback=deviation[split_index:])
    sad_together = _sad_fallback(center, areas=area, fallback=deviation)

    sum_right = area[:split_index].sum()
    sum_left = area[split_index:].sum()
    sum_together = sum_left + sum_right

    result = sad_left / sum_left**rescale_factor + sad_right / sum_right**rescale_factor
    result /= sad_together / sum_together**rescale_factor
    return 1 - result


@numba.jit(numba.void(numba.int64[:],
                      numba.float64[:], numba.float64[:], numba.float64[:],
                      numba.float64[:], numba.float64),
           nopython=True)
def compute_every_split_goodness(split_indices,
                                 center, deviation, area,
                                 results, rescale_factor):
    """Computes the "goodness of split" for several split points: see compute_split_goodness"""
    for i, split_i in enumerate(split_indices):
        # Index of hit to split on = index first hit that will go to right cluster
        results[i] = compute_split_goodness(split_i, center, deviation, area, rescale_factor)
