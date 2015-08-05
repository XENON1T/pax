import numpy as np

from pax import plugin, utils


class BasicProperties(plugin.TransformPlugin):
    """Computes basic peak properies such as area and hit time spread ("width")
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.last_top_ch = np.max(np.array(self.config['channels_top']))

    def transform_event(self, event):

        for peak in event.peaks:

            # For backwards compatibility with plotting code
            highest_peak_index = np.argmax([s.height for s in peak.hits])
            peak.index_of_maximum = peak.hits[highest_peak_index].index_of_maximum
            peak.height = peak.hits[highest_peak_index].height

            # Compute the area per pmt and saturation count
            peak.n_saturated_per_channel = np.zeros(event.n_channels, dtype=np.int16)
            for s in peak.hits:
                ch = s.channel
                peak.area_per_channel[ch] += s.area
                peak.n_saturated_per_channel[ch] += s.n_saturated

            # Compute the total area and saturation count
            peak.area = np.sum(peak.area_per_channel)
            peak.n_saturated = np.sum(peak.n_saturated_per_channel)

            # Compute top fraction
            peak.area_fraction_top = np.sum(peak.area_per_channel[:self.last_top_ch + 1]) / peak.area

            # Compute timing quantities
            times = [s.center for s in peak.hits]
            peak.hit_time_mean, peak.hit_time_std = utils.weighted_mean_variance(times,
                                                                                 [s.area for s in peak.hits])
            peak.hit_time_std **= 0.5  # Convert variance to std

            # Compute mean amplitude / noise
            if event.event_number == 198:
                pass
            hit_areas = [hit.area for hit in peak.hits]
            for hit in peak.hits:
                if hit.noise_sigma == 0:
                    pass
            try:
                peak.mean_amplitude_to_noise = np.average([hit.height / hit.noise_sigma for hit in peak.hits],
                                                          weights=hit_areas)
            except ZeroDivisionError:
                pass

            # Compute central ranges
            dt = event.sample_duration
            leftmost = float('inf')
            rightmost = float('-inf')
            area_so_far = 0
            for hit in sorted(peak.hits, key=lambda hit: abs(hit.center - peak.hit_time_mean)):
                if hit.left < leftmost:
                    leftmost = hit.left
                if hit.right > rightmost:
                    rightmost = hit.right
                area_so_far += hit.area
                if peak.range_50p_area == 0:
                    if area_so_far >= 0.5 * peak.area:
                        peak.range_50p_area = (rightmost - leftmost + 1) * dt
                if area_so_far >= 0.9 * peak.area:
                    peak.range_90p_area = (rightmost - leftmost + 1) * dt
                    break

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

                setattr(peak, '%s_hitpattern_spread' % array, np.sqrt(weighted_var))

        return event
