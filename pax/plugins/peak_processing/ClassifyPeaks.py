from pax import plugin, units
from scipy import interpolate


class AdHocClassification1T(plugin.TransformPlugin):

    def startup(self):
        self.s1_rise_time_bound = interpolate.interp1d([0, 5, 10, 100],
                                                       [100, 100, 70, 70],
                                                       fill_value='extrapolate', kind='linear')
        self.s1_rise_time_aft = interpolate.interp1d([0, 0.3, 0.4, 0.5, 0.70, 0.70, 1.0],
                                                     [70, 70, 65, 60, 35, 0, 0], kind='linear')

    def transform_event(self, event):

        for peak in event.peaks:
            # Don't work on noise and lone hit
            if peak.type in ('noise', 'lone_hit'):
                continue

            if -peak.area_decile_from_midpoint[1] < self.s1_rise_time_bound(peak.area):
                # Peak rises fast, could be S1
                if peak.tight_coincidence <= 2:
                    # Too few PMTs contributing, hard to distinguish from junk
                    peak.type = 'unknown'
                elif peak.area > 100 or peak.area < 5:
                    # Apply single electron s2 cut only in 5 - 100 PE range, otherwise only rely on rise time
                    peak.type = 's1'
                elif -peak.area_decile_from_midpoint[1] < self.s1_rise_time_aft(peak.area_fraction_top):
                    # Rise time and AFT as multi-demensional discriminator
                    peak.type = 's1'
                else:
                    peak.type = 's2'
            else:
                # No fast rise
                if peak.n_contributing_channels > 4:
                    # Large enough: can be S2
                    peak.type = 's2'
                else:
                    # Too few contributing channels, not really S2
                    peak.type = 'unknown'

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
