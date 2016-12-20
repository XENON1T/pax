from pax import plugin, units
import numpy as np
from scipy import interpolate


class AdHocClassification1T(plugin.TransformPlugin):

    def transform_event(self, event):

        if self.config.get('revert_to_old'):
            ##
            # OLD CLASSIFICATION, will be deleted soonish
            ##
            for peak in event.peaks:
                # Don't work on noise and lone_hit
                if peak.type in ('noise', 'lone_hit'):
                    continue

                area = peak.area
                width = peak.range_area_decile[5]

                if width > 0.1 * area**3:
                    peak.type = 'unknown'

                elif width < 50:
                    peak.type = 's1'

                else:
                    if width > 3.5e2 / area**0.1 or width > 1.5 * area:
                        peak.type = 's2'
                    else:
                        peak.type = 's1'

        else:
            ##
            # NEW CLASSIFICATION
            ##
            min_s2_aft = 0.3

            x_s1 = np.array([0, 40,  40 + 1e-9,  70, 120, 121])
            y_s1 = np.array([1,  1,  0.7,      0.3, 0,     0])
            s1_classification_bound = interpolate.interp1d(x_s1, y_s1, fill_value='extrapolate')
            s2_classification_bound = interpolate.interp1d(x_s1 + 20,
                                                           np.clip(y_s1, min_s2_aft, 1),
                                                           fill_value='extrapolate')

            # Sort the peaks by left boundary, so we can keep track of the largest peak seen so far
            event.peaks = sorted(event.peaks, key=lambda x: x.left)

            largest_peak_area_seen = 0

            for peak in event.peaks:
                # Don't work on noise and lone_hit
                if peak.type in ('noise', 'lone_hit'):
                    continue

                # Peaks with a low coincidence level are labeled 'unknown' immediately
                if peak.n_contributing_channels <= 3:
                    peak.type = 'unknown'
                    continue

                area = peak.area
                width = peak.range_area_decile[5]
                aft = peak.area_fraction_top

                if area < 50:
                    # Decide on S1/S2 based on area fraction top and width.
                    # For widths < 40 ns, always S1; > 120 ns, always S2. In between it depends also on aft.
                    if aft < s1_classification_bound(width):
                        if largest_peak_area_seen > 5000:
                            # We're already in the tail of a large peak. If we see an S1-like peak here, it is most
                            # likely a (fragment of) a small single electron
                            peak.type = 'unknown'
                        else:
                            peak.type = 's1'
                    elif aft > s2_classification_bound(width):
                        peak.type = 's2'
                    else:
                        peak.type = 'unknown'

                else:
                    if width < 70 or (width < 3.5e2 / area**0.1 and width < 1.5 * area):
                        peak.type = 's1'
                    else:
                        if aft < min_s2_aft:
                            peak.type == 'unknown'
                        else:
                            peak.type = 's2'

                largest_peak_area_seen = max(largest_peak_area_seen, area)

        return event


class AdHocClassification(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:
            # Don't work on noise and lone_hit
            if peak.type in ('noise', 'lone_hit'):
                continue

            width = peak.range_area_decile[5]

            if peak.area > 50:
                # We don't have to worry about single electrons anymore
                if width < 100 * units.ns:
                    peak.type = 's1'
                elif width > 250 * units.ns:
                    peak.type = 's2'
            else:
                # Worry about SE-S1 identification.
                if width < 75 * units.ns:
                    peak.type = 's1'
                else:
                    if peak.area < 5:
                        peak.type = 'coincidence'
                    elif width > 100 * units.ns:
                        peak.type = 's2'

        return event
