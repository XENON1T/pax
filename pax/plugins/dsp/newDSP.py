import numpy as np
import math
from pax import plugin, datastructure, dsputils


class FindPeaks(plugin.TransformPlugin):
    # Find peaks in intervals above threshold in filtered waveform
    # TODO: finds some 'half-peaks' where max = bound ...???

    def transform_event(self, event):
        for pf in self.config['peakfinders']:
            peakfinding_wave = event.get_waveform(pf['peakfinding_wave']).samples
            unfiltered = event.get_waveform(pf['unfiltered_wave']).samples
            peaks = []

            # Find regions currently free of peaks
            if len(event.peaks) == 0 or ('ignore_previous_peaks' in pf and pf['ignore_previous_peaks']):
                pf_regions = [(0, len(peakfinding_wave) - 1)]
            else:
                pf_regions = dsputils.free_regions(event)

            # Search for peaks in the free regions
            for region_left, region_right in pf_regions:
                for itv_left, itv_right in dsputils.intervals_above_threshold(
                        peakfinding_wave[region_left:region_right + 1], pf['threshold']):
                    
                    # Find the left & right peak bounds in the peakfinding wave
                    signal = peakfinding_wave[region_left + itv_left : region_left + itv_right + 1]
                    left, right = dsputils.peak_bounds(signal, np.argmax(signal), pf['peak_integration_bound'])

                    # Find the index of maximum and height in the unfiltered wave
                    unfiltered_signal = unfiltered[region_left + itv_left + left : region_left + itv_left + right + 1]
                    unfiltered_max_idx = region_left + itv_left + left + np.argmax(unfiltered_signal)
                    p = datastructure.Peak({
                        'index_of_maximum': unfiltered_max_idx,
                        'height':           unfiltered[unfiltered_max_idx],
                        'left':             region_left + itv_left + left,
                        'right':            region_left + itv_left + right,
                    })
                    # Should we already label the peak?
                    if 'force_peak_label' in pf:
                        p.type = pf['force_peak_label']
                    peaks.append(p)

                    # Recursion on leftover intervals is not worth it: peaks 100x as small are boring.
                    # TODO: this was wrong, but can't remember why.. think!
                    # Well... double scatters? No, can't distinguish from photo-ionizations when they're this low

            # Peaks no longer overlap now that we've enabled constrain_bounds.

            self.log.debug("Found %s peaks in %s." % (len(peaks), pf['peakfinding_wave']))
            event.peaks.extend(peaks)

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




class ComputePeakAreas(plugin.TransformPlugin):

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
                if self.config['exlude_non_contributing_channels_from_area']:
                    peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])
                else:
                    peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])

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
            left_samples = math.floor(self.config['s1_half_area_in']/2)
            right_samples = math.ceil(self.config['s1_half_area_in']/2)
            if np.sum(unfiltered[p.index_of_maximum - left_samples: p.index_of_maximum + right_samples]) > 0.5 * p.area:
                p.type = 's1'
                #self.log.debug("%s-%s-%s: S1" % (p.left, p.index_of_maximum, p.right))
            else:
                p.type = 's2'
                #self.log.debug("%s-%s-%s: S2" % (p.left, p.index_of_maximum, p.right))
        return event
