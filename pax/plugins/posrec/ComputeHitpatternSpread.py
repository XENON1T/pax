import numpy as np

from pax import plugin


class ComputeHitpatternSpread(plugin.TransformPlugin):

    """Computes the weighted root mean square deviation of the top and bottom hitpattern for each peak
    """

    def startup(self):

        # grab PMT numbers and x, y locations in each array
        self.pmts = {}
        self.locations = {}
        for array in ('top', 'bottom'):
            self.pmts[array] = self.config['channels_%s' % array]
            self.locations[array] = {}
            for dim in ('x', 'y'):
                self.locations[array][dim] = np.array([self.config['pmt_locations'][ch][dim]
                                                       for ch in self.pmts[array]])

    def transform_event(self, event):

        for peak in event.peaks:

            # No point in computing this for veto peaks
            if peak.detector != 'tpc':
                continue

            for array in ('top', 'bottom'):

                hitpattern = peak.area_per_channel[self.pmts[array]]

                if np.all(hitpattern == 0.0):
                    # Empty hitpatterns will give error in np.average
                    continue

                weighted_var = 0
                for dim in ('x', 'y'):
                    weighted_var += weighted_variance(self.locations[array][dim],
                                                      weights=hitpattern)

                setattr(peak, 'hitpattern_%s_spread' % array, np.sqrt(weighted_var/2))

        return event


def weighted_variance(values, weights):
    """
    Return the weighted sum square deviation from the weighted mean.
    values, weights -- Numpy ndarrays with the same shape.
    Stolen from http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    weighted_average = np.average(values, weights=weights)
    variance = np.average((values-weighted_average)**2, weights=weights)  # Fast and numerically precise
    return variance
