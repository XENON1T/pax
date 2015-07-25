import numpy as np

from pax import plugin

from pax.datastructure import ReconstructedPosition


class PosRecRobustWeightedMean(plugin.TransformPlugin):
    """Reconstruct S2 positions using an iterative weighted mean algorithm. For each S2:
    1. Compute area-weighted mean of top hitpattern
    2. Compute the area-weighted mean distance of all remaining PMTs from this position.
    3. Remove PMTs which are 'outliers': further than outlier_threshold * mean distance from the weighted mean position
       If all PMTs are outliers (can happen if hitpattern is really bizarre), only the furthest one is removed
    This is repeated until step 3 removes no outliers, or there are less than min_pmts_left PMTs left after step 3.
    """

    def startup(self):
        self.outlier_threshold = self.config['outlier_threshold']

        # List of integers of which PMTs to use, this algorithm uses the top pmt array to reconstruct
        self.pmts = self.config['channels_top']

        # (x,y) Locations of these PMTs, stored as np.array([(x,y), (x,y), ...])
        self.pmt_locations = np.zeros((len(self.pmts), 2))
        for ch in self.pmts:
            for dim in ('x', 'y'):
                self.pmt_locations[ch][{'x': 0, 'y': 1}[dim]] = self.config['pmt_locations'][ch][dim]

    def transform_event(self, event):

        for peak in event.S2s():

            # Start with all PMTs in self.pmts that have a hit
            # Could use peak.contributing_channels, but this is calculated in ComputePeakProperties
            # and then would have to adapt PosRecTest. Or we could make does_channel_contribute a property...
            pmts = np.intersect1d(np.where(peak.area_per_channel > 0)[0],
                                  self.pmts)

            if len(pmts) <= 1:
                # How on earth did this get classified as S2??
                wmp = [float('nan'), float('nan')]

            while True:

                # Get locations and hitpattern of remaining PMTs
                pmt_locs = self.pmt_locations[pmts]
                hitpattern = peak.area_per_channel[pmts]

                # Compute the weighted mean position (2-vector)
                wmp = np.average(pmt_locs, weights=hitpattern, axis=0)

                # Compute the Euclidean distance between PMTs and the wm position
                distances = np.sum((wmp[np.newaxis, :] - pmt_locs)**2, axis=1)**0.5

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
                if len(pmts) <= self.config['min_pmts_left']:
                    break

            peak.reconstructed_positions.append(ReconstructedPosition(x=wmp[0], y=wmp[1],
                                                                      algorithm=self.name))
        return event
