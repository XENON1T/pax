import numpy as np
from pax import plugin, utils


class ComputePeakProperties(plugin.TransformPlugin):

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
