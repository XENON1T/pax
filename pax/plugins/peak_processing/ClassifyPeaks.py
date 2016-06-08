from pax import plugin, units


class AdHocClassification1T(plugin.TransformPlugin):

    def transform_event(self, event):

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
