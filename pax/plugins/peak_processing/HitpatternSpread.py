import numpy as np
from pax import plugin


class HitpatternSpread(plugin.TransformPlugin):
    """Computes the weighted root mean square deviation of the top and bottom hitpattern for each peak
    """

    def startup(self):

        # Grab PMT numbers and x, y locations in each array
        self.pmts = {}
        self.locations = {}
        for array in ('top', 'bottom'):
            self.pmts[array] = self.config['channels_%s' % array]
            self.locations[array] = np.zeros((len(self.pmts[array]), 2))
            for i, ch in enumerate(self.pmts[array]):
                for dim in ('x', 'y'):
                    self.locations[array][i][{'x': 0, 'y': 1}[dim]] = self.config['pmt_locations'][ch][dim]

    def transform_event(self, event):

        for peak in event.peaks:

            # No point in computing this for veto peaks
            if peak.detector != 'tpc':
                continue

            for array in ('top', 'bottom'):

                hitpattern = peak.area_per_channel[self.pmts[array]]

                if np.all(hitpattern == 0):
                    # Empty hitpatterns will give error in np.average
                    continue

                # Compute the weighted mean position
                weighted_mean_position = np.average(self.locations[array], weights=hitpattern, axis=0)

                # Compute weighted average euclidean distance from mean position
                avg_distance = np.sqrt(np.average(np.sum((self.locations[array] - weighted_mean_position) ** 2, axis=1),
                                                  weights=hitpattern))

                setattr(peak, '%s_hitpattern_spread' % array, avg_distance)

        return event
