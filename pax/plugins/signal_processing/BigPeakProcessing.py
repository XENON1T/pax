"""
Plugins for computing properties of peaks that have been found
"""
import numpy as np
from pax import utils, plugin
import math


class DeleteSmallPeaks(plugin.TransformPlugin):

    """Deletes low coincidence peaks, so the low-energy peakfinder can have a crack at them"""

    def transform_event(self, event):
        event.peaks = [p for p in event.peaks
                       if p.number_of_contributing_channels >= self.config['prune_if_coincidence_lower_than']
                       and p.area >= self.config['prune_if_area_lower_than']]
        return event


class ComputePeakWidths(plugin.TransformPlugin):

    """Does what it says on the tin"""

    def transform_event(self, event):
        for peak in event.peaks:

            # Check if peak is sane
            if peak.index_of_maximum < peak.left:
                self.log.debug("Insane peak %s-%s-%s, can't compute widths!" % (
                    peak.left, peak.index_of_maximum, peak.right))
                continue

            for width_name, conf in self.config['width_computations'].items():

                peak[width_name] = utils.width_at_fraction(
                    peak_wave=event.get_sum_waveform(conf['waveform_to_use']).samples[peak.left: peak.right + 1],
                    fraction_of_max=conf['fraction_of_max'],
                    max_idx=peak.index_of_maximum - peak.left,
                    interpolate=conf['interpolate'])

        return event


class ComputePeakAreasAndCoincidence(plugin.TransformPlugin):

    def startup(self):
        self.central_width = round(self.config['central_area_region_width'] / self.config['sample_duration'], 1)

    def transform_event(self, event):
        for peak in event.peaks:

            # Compute area in each channel
            # Note this also computes the area for PMTs in other detectors!
            peak.area_per_channel = np.sum(event.channel_waveforms[:, peak.left:peak.right + 1], axis=1)

            # Determine which channels contribute to the peak's total area
            channels = np.arange(self.config['n_channels'])
            peak.does_channel_contribute = np.in1d(channels,
                                                   np.array(list(self.config['channels_in_detector'][peak.detector]))) & \
                (peak.area_per_channel >= self.config['minimum_area'])

            # Other channels with nonzero area have noise
            peak.does_channel_have_noise = np.in1d(channels,
                                                   np.array(list(self.config['channels_in_detector'][peak.detector]))) & \
                (peak.area_per_channel != 0) & \
                (np.invert(peak.does_channel_contribute))

            # Compute the peak's area
            peak.area = np.sum(peak.area_per_channel[peak.contributing_channels])

            # Compute the peak's central area (used for classification)
            peak.central_area = np.sum(
                event.channel_waveforms[
                    peak.contributing_channels,
                    max(peak.left, peak.index_of_maximum - math.floor(self.central_width / 2)):
                    min(peak.right + 1, peak.index_of_maximum + math.ceil(self.central_width / 2))
                ],
                axis=(0, 1)
            )

        return event


class ComputePeakEntropies(plugin.TransformPlugin):
    # TODO: write tests

    def transform_event(self, event):
        for peak in event.peaks:

            peak_waveforms = event.channel_waveforms[:, peak.left:peak.right + 1]
            if self.config['normalization_mode'] is 'abs':
                normalized = np.abs(peak_waveforms)
            elif self.config['normalization_mode'] is 'square':
                normalized = peak_waveforms ** 2
            else:
                raise ValueError(
                    'Invalid Configuration for ComputePeakEntropies: normalization_mode must be abs or square')

            # In the case of abs, we could re-use peak.area_per_channel to normalize
            # This gains only a little bit of performance, and 'square' is what we use in Xenon100 anyway.
            # Note the use of np.newaxis to enable numpy broadcasting of the division
            normalized /= peak.area_per_channel[:, np.newaxis]

            # Could this be vectorized better?
            # Restricting to contributing pmts is not optional: otherwise you'd include other detectors as well.
            peak.entropy_per_channel = np.zeros(len(peak_waveforms))
            for pmt in peak.contributing_channels:
                peak.entropy_per_channel[pmt] = -np.sum(normalized[pmt] * np.log(normalized[pmt]))

        return event


class ClassifyBigPeaks(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:
            # PLACEHOLDER:
            # if central area is > central_area_ratio * total area, christen as S1 candidate
            if peak.central_area > self.config['central_area_ratio'] * peak.area:
                peak.type = 's1'
            else:
                peak.type = 's2'
        return event
