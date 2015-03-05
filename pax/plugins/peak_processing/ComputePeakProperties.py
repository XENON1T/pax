import numpy as np
from pax import plugin, utils


class Basics(plugin.TransformPlugin):

    """Computes basic peak properies such as area and hit time spread ("width")
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.last_top_ch = np.max(np.array(self.config['channels_top']))

    def transform_event(self, event):

        for peak in event.peaks:

            peak.left = min([s.left for s in peak.channel_peaks])
            peak.right = max([s.right for s in peak.channel_peaks])

            # For backwards compatibility with plotting code
            highest_peak_index = np.argmax([s.height for s in peak.channel_peaks])
            peak.index_of_maximum = peak.channel_peaks[highest_peak_index].index_of_maximum
            peak.height = peak.channel_peaks[highest_peak_index].height

            # Compute the area per pmt. Store maxidx as well
            for s in peak.channel_peaks:
                peak.area_per_channel[s.channel] += s.area

            # Compute the total area
            peak.area = np.sum(peak.area_per_channel)

            # Compute top fraction
            peak.area_fraction_top = np.sum(peak.area_per_channel[:self.last_top_ch + 1])

            # Compute timing quantities
            times = [s.index_of_maximum * self.dt for s in peak.channel_peaks]
            peak.median_absolute_deviation = utils.mad(times)
            peak.hit_time_mean, peak.hit_time_std = utils.weighted_mean_variance(times,
                                                                                 [s.area for s in peak.channel_peaks])
            peak.hit_time_std **= 0.5   # We stored variance in the line above

        return event


class HitpatternSpread(plugin.TransformPlugin):

    """Computes the weighted root mean square deviation of the top and bottom hitpattern for each peak
    """

    def startup(self):

        # Grab PMT numbers and x, y locations in each array
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
                    _, wv = utils.weighted_mean_variance(self.locations[array][dim],
                                                         weights=hitpattern)
                    weighted_var += wv

                setattr(peak, '%s_hitpattern_spread' % array, np.sqrt(weighted_var/2))

        return event
