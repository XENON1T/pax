import numpy as np
from pax import plugin


class PosRecRobustWeightedMean(plugin.PosRecPlugin):
    """Reconstruct S2 positions using an iterative weighted mean algorithm. For each S2:
    1. Compute area-weighted mean of top hitpattern
    2. Compute the area-weighted mean distance of all remaining PMTs from this position.
    3. Remove PMTs which are 'outliers': further than outlier_threshold * mean distance from the weighted mean position
       If all PMTs are outliers (can happen if hitpattern is really bizarre), only the furthest one is removed
    This is repeated until step 3 removes no outliers, or there are less than min_pmts_left PMTs left after step 3.
    """

    def startup(self):
        self.outlier_threshold = self.config['outlier_threshold']

        self.outer_ring_pmts = self.config['outer_ring_pmts']
        self.outer_ring_multiplication_factor = self.config.get('outer_ring_multiplication_factor', 1)

    def reconstruct_position(self, peak):
        # Upweigh the outer ring to compensate for their partial obscuration by the TPC wall
        area_per_channel = peak.area_per_channel.copy()
        area_per_channel[self.outer_ring_pmts] *= self.outer_ring_multiplication_factor

        # Start with all PMTs in self.pmts that have some area
        pmts = np.intersect1d(np.where(area_per_channel > 0)[0],
                              self.pmts)

        if len(pmts) == 1:
            return self.pmt_locations[pmts[0]]

        while True:
            # Get locations and hitpattern of remaining PMTs
            pmt_locs = self.pmt_locations[pmts]
            hitpattern = area_per_channel[pmts]

            # Rare case where somehow no pmts are contributing??
            if np.sum(hitpattern) == 0:
                break

            # Compute the weighted mean position (2-vector)
            weighted_mean_position = np.average(pmt_locs, weights=hitpattern, axis=0)

            # Compute the Euclidean distance between PMTs and the wm position
            distances = np.sum((weighted_mean_position[np.newaxis, :] - pmt_locs)**2, axis=1)**0.5

            # Compute the weighted mean distance
            wmd = np.average(distances, weights=hitpattern)

            # If there are no outliers, we are done
            is_outlier = distances > wmd * self.outlier_threshold
            if not np.any(is_outlier):
                break

            if np.all(is_outlier):
                # All are outliers... remove just the worst
                pmts = np.delete(pmts, np.argmax(distances))
            else:
                # Remove all outliers
                pmts = pmts[True ^ is_outlier]

            # Give up if there are too few PMTs left
            # Don't put this as the while condition, want loop to run at least once
            if len(pmts) <= self.config['min_pmts_left']:
                break

        return weighted_mean_position
