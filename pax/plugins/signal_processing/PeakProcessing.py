"""
Plugins for computing properties of peaks that have been found
"""
import numpy as np
from pax import dsputils, plugin
import math


class DeleteSmallPeaks(plugin.TransformPlugin):
    """Deletes low coincidence peaks, so the low-energy peakfinder can have a crack at them"""
    def transform_event(self, event):
        event.peaks = [p for p in event.peaks
                       if p.coincidence_level >= self.config['prune_if_coincidence_lower_than']
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

                peak[width_name] = dsputils.width_at_fraction(
                    peak_wave=event.get_waveform(conf['waveform_to_use']).samples[peak.left : peak.right+1],
                    fraction_of_max=conf['fraction_of_max'],
                    max_idx=peak.index_of_maximum - peak.left,
                    interpolate=conf['interpolate'])

        return event




class ComputePeakAreasAndCoincidence(plugin.TransformPlugin):

    def transform_event(self, event):
        for peak in event.peaks:

            # Compute area in each channel
            peak.area_per_pmt = np.sum(event.pmt_waveforms[:, peak.left:peak.right+1], axis=1)

            # Determine which channels contribute to the peak's total area
            peak.contributing_pmts = np.array(
                np.where(peak.area_per_pmt >= self.config['minimum_area'])[0],
                dtype=np.uint16)

            # Compute the peak's areas
            # TODO: make excluding non-contributing pmts optional
            if peak.type == 'veto':
                peak.area = np.sum(peak.area_per_pmt[list(self.config['pmts_veto'])])
            else:
                if self.config['exclude_non_contributing_channels_from_area']:
                    peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])
                else:
                    peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])

        return event


class ComputePeakEntropies(plugin.TransformPlugin):
    #TODO: write tests


    def transform_event(self, event):
        for peak in event.peaks:

            peak_waveforms = event.pmt_waveforms[:, peak.left:peak.right+1]

            # Switching from entropy to kurtosis doesn't make it faster...
            # At head put:
            # import scipy
            # from scipy import stats
            # Here put:
            # peak.entropy_per_pmt = scipy.stats.kurtosis(peak_waveforms, axis=1)
            # continue

            if self.config['normalization_mode'] is 'abs':
                normalized = np.abs(peak_waveforms)
            elif self.config['normalization_mode'] is 'square':
                normalized = peak_waveforms**2
            else:
                raise ValueError(
                    'Invalid Configuration for ComputePeakEntropies: normalization_mode must be abs or square')

            # In the case of abs, we could re-use peak.area_per_pmt to normalize
            # This gains only a little bit of performance, and 'square' is what we use in Xenon100 anyway.
            # Note the use of np.newaxis to enable numpy broadcasting of the division
            normalized /= peak.area_per_pmt[:, np.newaxis]

            if self.config['only_for_contributing_pmts']:
                # Could this be vectorized better?
                # There is probably little use in restricting to a set of pmts before here,
                # the logarithm contains most of the work.
                peak.entropy_per_pmt = np.zeros(len(peak_waveforms))
                for pmt in peak.contributing_pmts:
                    peak.entropy_per_pmt[pmt] = -np.sum(normalized[pmt]*np.log(normalized[pmt]))
            else:
                peak.entropy_per_pmt = -np.sum(normalized*np.log(normalized), axis=1)

        return event



class IdentifyPeaks(plugin.TransformPlugin):

    def transform_event(self, event):

        unfiltered = event.get_waveform('tpc').samples
        for p in event.peaks:
            if p.type != 'unknown':
                # Some peakfinder forced the type. Fine, not my problem...
                continue
            # PLACEHOLDER:
            # if area in s1_half_area_in samples around max is > 50% of total area, christen as S1 candidate
            # if peak is smaller than s1_half_area_in, it is certainly an s1
            if p.right - p.left + 1 < self.config['s1_half_area_in']:
                p.type = 's1'
            else:
                left_samples = math.floor(self.config['s1_half_area_in']/2)
                right_samples = math.ceil(self.config['s1_half_area_in']/2)
                if np.sum(unfiltered[p.index_of_maximum - left_samples: p.index_of_maximum + right_samples]) > 0.5 * p.area:
                    p.type = 's1'
                    #self.log.debug("%s-%s-%s: S1" % (p.left, p.index_of_maximum, p.right))
                else:
                    p.type = 's2'
                    #self.log.debug("%s-%s-%s: S2" % (p.left, p.index_of_maximum, p.right))
        return event



# class PruneSmallPeaksNearLargePeaks(plugin.TransformPlugin):
#     """Delete small peaks too close to a large peak
#     Large peaks trigger many channels, so the noise in the sum waveform noise becomes large
#     (if the noise is gaussian, it scales as ~sqrt(nchannels).
#     The low-energy peakfinder should take care of fake peaks in these regions just fine,
#     but it is expensive, so if we are certain a peak can't be real, might as well delete it already.
#     """
#
#     def transform_event(self, event):
#         large_peaks = [p for p in event.peaks if p.area >= self.config['large_peaks_start_from']]
#         small_peaks = [p for p in event.peaks if p.area < self.config['always_keep_peaks_larger_than']]
#         print(large_peaks)
#         for p in small_peaks:
#             if p.type in self.config['never_prune_peak_types']:
#                 continue
#
#             largepeakstoleft = [q for q in large_peaks if q.left < p.left]
#             if len(largepeakstoleft) > 0:
#                 leftlargepeak = max(largepeakstoleft, key=lambda x : x.left)
#                 if p.left - leftlargepeak.right < self.config['min_distance_to_large_peak']:
#                     self.log.warning("Prune %s at %s-%s-%s: too close to large peak on the left" % (
#                         p.type, p.left, p.index_of_maximum, p.right))
#                     p.type = 'X'
#                     continue
#
#             largepeakstoright = [q for q in large_peaks if q.right > p.right]
#             if len(largepeakstoright) > 0:
#                 rightlargepeak = min(largepeakstoright, key=lambda x : x.right)
#                 if rightlargepeak.left - p.right < self.config['min_distance_to_large_peak']:
#                     self.log.warning("Prune %s at %s-%s-%s: too close to large peak on the right" % (
#                         p.type, p.left, p.index_of_maximum, p.right))
#                     p.type = 'X'
#                     continue
#
#         #event.peaks = [p for p in event.peaks if p.type != 'X']
#         return event
