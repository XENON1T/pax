from pax import plugin, units
from scipy import interpolate


class AdHocClassification1T(plugin.TransformPlugin):

    def startup(self):
        self.s1_rise_time_bound = interpolate.interp1d([0, 10, 25, 100],
                                                       [100, 100, 70, 70],
                                                       fill_value='extrapolate', kind='linear')

    def transform_event(self, event):

        for peak in event.peaks:
            # Don't work on noise and lone_hit
            if peak.type in ('noise', 'lone_hit'):
                continue

            # Peaks with a low coincidence level are labeled 'unknown' immediately
            if peak.tight_coincidence <= 2:
                peak.type = 'unknown'
                continue

            # Peaks that rise fast are S1s, the rest are S2s
            if -peak.area_decile_from_midpoint[1] < self.s1_rise_time_bound(peak.area):
                peak.type = 's1'
            else:
                peak.type = 's2'

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
