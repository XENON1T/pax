import numpy as np
from pax import plugin, datastructure, dsputils


class FindPeaks(plugin.TransformPlugin):
    # Find peaks in intervals above threshold in filtered waveform

    def transform_event(self, event):
        for pf in self.config['peakfinders']:
            filtered = event.get_waveform(pf['peakfinding_wave']).samples
            unfiltered = event.get_waveform(pf['unfiltered_wave']).samples
            peaks = []

            # Find regions currently free of peaks
            if len(event.peaks) == 0 or ('ignore_previous_peaks' in pf and pf['ignore_previous_peaks']):
                pf_regions = [(0, len(filtered) - 1)]
            else:
                pf_regions = dsputils.free_regions(event)

            # Search for peaks in the free regions
            for region_left, region_right in pf_regions:
                region_filtered = filtered[region_left:region_right + 1]
                region_unfiltered = unfiltered[region_left:region_right + 1]
                for itv_left, itv_right in dsputils.intervals_above_threshold(region_filtered, pf['threshold']):
                    p = dsputils.find_peak_in_signal(
                        signal=region_filtered[itv_left : itv_right + 1],
                        unfiltered=region_unfiltered[itv_left : itv_right + 1],
                        integration_bound_fraction=pf['peak_integration_bound'],
                        offset=region_left + itv_left,
                    )
                    # Reursion on leftover intervals is not worth it: peaks 100x as small are boring.
                    # Well... double scatters? No, can't distinguish from photo-ionizations when they're this low

                    # Should we already label the peak?
                    if 'force_peak_label' in pf:
                        p.type = pf['force_peak_label']
                    peaks.append(p)

            # Peaks no longer overlap now that we've enabled constrain_bounds.

            self.log.debug("Found %s peaks in %s." % (len(peaks), pf['peakfinding_wave']))
            event.peaks.extend(peaks)
        return event



class IdentifyPeaks(plugin.TransformPlugin):

    def transform_event(self, event):

        unfiltered = event.get_waveform('tpc').samples
        for p in event.peaks:
            if p.type != 'unknown':
                # Some peakfinder forced the type. Fine, not my problem...
                continue
            # PLACEHOLDER:
            # if area in 5 samples around max i s > 50% of total area, christen as S1 candidate
            if np.sum(unfiltered[p.index_of_maximum - 2: p.index_of_maximum + 3]) > 0.5 * p.area:
                p.type = 's1'
                #self.log.debug("%s-%s-%s: S1" % (p.left, p.index_of_maximum, p.right))
            else:
                p.type = 's2'
                #self.log.debug("%s-%s-%s: S2" % (p.left, p.index_of_maximum, p.right))
        return event


class ComputePeakProperties(plugin.TransformPlugin):

    def transform_event(self, event):
        dt = self.config['digitizer_t_resolution']
        for peak in event.peaks:
            peak.area_per_pmt = np.sum(event.pmt_waveforms[:, peak.left:peak.right], axis=1)
            veto_area = np.sum(peak.area_per_pmt[list(self.config['pmts_veto'])])
            if peak.type == 'veto':
                peak.area = veto_area
            else:
                peak.area = np.sum(peak.area_per_pmt) - veto_area
            #Todo: exclude negative area channels if option given
            peak.coincidence_level = len(np.where(peak.area_per_pmt >= self.config['minimum_area'])[0])
            if not 'keep_Chris_happy' in self.config or self.config['keep_Chris_happy'] == True:
                continue
            # Dynamically set lots of event class attributes
            for waveform in event.waveforms:
                peak_wave = waveform.samples[peak.left : peak.right]
                secretly_set_attribute(peak, waveform.name + '_' + 'area',           np.sum(peak_wave))
                max_idx = np.argmax(peak_wave)
                secretly_set_attribute(peak, waveform.name + '_' + 'argmax',         peak.left + max_idx)
                secretly_set_attribute(peak, waveform.name + '_' + 'height',         peak_wave[max_idx])
                secretly_set_attribute(peak, waveform.name + '_' + 'inferred_width',
                                       dt * 2 * getattr(peak, waveform.name + '_' + 'area') / getattr(peak, waveform.name + '_' + 'height')
                )
                # Width computations are expensive... comment these out if you're in a rush
                for name, value in (('fwhm', 0.5), ('fwqm', 0.25), ('fwtm', 0.1)):
                    secretly_set_attribute(peak, waveform.name + '_' + name,
                        dsputils.width_at_fraction(peak_wave, fraction_of_max=value,  max_idx=max_idx) * dt
                    )
        return event

def secretly_set_attribute(object, name, value):
    from pax.micromodels.fields import IntegerField, FloatField
    object.__setattr__(object, name, FloatField())
    object.__setattr__(object, name, value)


class SplitPeaks(plugin.TransformPlugin):

    def startup(self):
        def is_valid_p_v_pair(signal, peak, valley):
            return (
                abs(peak - valley) >= self.config['min_p_v_distance'] and
                signal[peak] / signal[valley] >= self.config['min_p_v_ratio'] and
                signal[peak] - signal[valley] >= self.config['min_p_v_difference']
            )
        self.is_valid_p_v_pair = is_valid_p_v_pair

    def transform_event(self, event):
        # TODO: this works on all peaks, but takes tpc and tpc_s2 as signals...
        filtered = event.get_waveform('tpc_s2').samples
        unfiltered = event.get_waveform('tpc').samples
        revised_peaks = []
        for parent in event.peaks:
            # If the peak is not large enough, it will not be split
            if ('composite_peak_min_width' in self.config and
                        parent.right - parent.left < self.config['composite_peak_min_width']
                    ):
                revised_peaks.append(parent)
                continue
            # Try to split the peak
            ps, vs = dsputils.peaks_and_valleys(
                filtered[parent.left:parent.right + 1],
                test_function=self.is_valid_p_v_pair
            )
            # If the peak wasn't split, we don't have to do anything
            if len(ps) < 2:
                revised_peaks.append(parent)
                continue

            # import matplotlib.pyplot as plt
            # plt.plot(event.get_waveform('tpc').samples[parent.left:parent.right+1])
            # plt.plot(filtered[parent.left:parent.right+1])
            # plt.plot(ps, filtered[parent.left + np.array(ps)], 'or')
            # plt.plot(vs, filtered[parent.left + np.array(vs)], 'ob')
            # plt.show()

            ps += parent.left
            vs += parent.left
            self.log.debug("S2 at " + str(parent.index_of_maximum) + ": peaks " + str(ps) + ", valleys " + str(vs))
            # Compute basic quantities for the sub-peaks
            for i, p in enumerate(ps):
                l_bound = vs[i - 1] if i != 0 else parent.left
                r_bound = vs[i]
                max_idx = l_bound + np.argmax(unfiltered[l_bound:r_bound + 1])
                new_peak = datastructure.Peak({
                    'index_of_maximum': max_idx,
                    'height':           unfiltered[max_idx],
                })
                # No need to recompute peak bounds: the whole parent peak is <0.01 max of the biggest peak
                # If we ever need to anyway, this code works:
                # left, right = dsputils.peak_bounds(filtered[l_bound:r_bound+1], max_i, 0.01)
                # new_peak.left  = left + l_bound
                # new_peak.right = right + l_bound
                new_peak.left = l_bound
                new_peak.right = r_bound
                revised_peaks.append(new_peak)
                new_peak.area = np.sum(unfiltered[new_peak.left:new_peak.right + 1])

        event.peaks = revised_peaks
        return event
