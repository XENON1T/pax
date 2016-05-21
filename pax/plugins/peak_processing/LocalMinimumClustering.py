from pax import plugin, datastructure
import numba
import logging
log = logging.getLogger('LocalMinimumClusteringHelpers')


class LocalMinimumClustering(plugin.ClusteringPlugin):

    def startup(self):
        import logging
        self.log.setLevel(logging.DEBUG)

    def cluster_peak(self, peak):
        # Unfortunate code duplication with BasicProperties...
        peak.left = peak.hits['left'].min()
        peak.right = peak.hits['right'].max()

        if peak.type == 'lone_hit':
            return [peak]
        w = self.event.get_sum_waveform(peak.detector).samples[peak.left:peak.right + 1]
        split_points = list(find_split_points(w,
                                              min_height=self.config.get('min_height', 15),
                                              min_ratio=self.config.get('min_ratio', 1.5)))
        if not len(split_points):
            return [peak]
        else:
            self.log.debug("Splitting %d-%d into %d peaks" % (peak.left, peak.right, len(split_points) + 1))
            return list(split_peak(peak, split_points))


def split_peak(peak, split_points):
    """Yields new peaks split from peak at split_points = sample indices within peak.
    Hits are distributed by the location of their maximum; if maximum is at a split point, hit will go to right.
    """
    # Iterate over left, right bounds of the new peaks
    for l, r in zip([0] + split_points, split_points + [peak.right]):
        l += peak.left
        r += peak.left

        # Select hits which have their maximum within this peak bounds
        # The last new peak must also contain hits at the right bound (though this is unlikely to happen)
        hits = peak.hits[(peak.hits['index_of_maximum'] >= l) &
                         (peak.hits['index_of_maximum'] < (float('inf') if r == peak.right else r))]

        if not len(hits):
            log.warning("Attempt to create a peak without hits (edge case of LocalMinimumClustering: no peak created!")
            continue

        yield datastructure.Peak(left=l, right=r,
                                 detector=peak.detector, hits=hits)


#@numba.jit(numba.float64[:])
def find_split_points(w, min_height, min_ratio):
    """"Finds local minima in w,
    whose peaks to the left and right both satisfy:
      - larger than minimum + min_height
      - larger than minimum * min_ratio
    """
    last_max = 0
    min_since_max = float('inf')
    min_since_max_i = 0

    for i, x in enumerate(w):
        if x > last_max:
            # New max, reset minimum finder state
            last_max = x
            min_since_max = float('inf')
            min_since_max_i = i

        if x < min_since_max:
            # New minimum since last max
            min_since_max = x
            min_since_max_i = i

        if min(last_max, x) > max(min_since_max + min_height,
                                  min_since_max * min_ratio):
            # Significant local minimum: tell caller, reset both max and min finder
            yield min_since_max_i
            last_max = x
            min_since_max = float('inf')
            min_since_max_i = i
