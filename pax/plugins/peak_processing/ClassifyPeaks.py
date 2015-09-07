from pax import plugin, units


class AdHocClassification(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:
            width = peak.range_area_decile[9]

            # Don't work on noise and lone_hit
            if peak.type in ('noise', 'lone_hit'):
                continue

            if width < 150 * units.ns:
                peak.type = 's1'

            elif width > 200 * units.ns:
                if peak.area > 5:
                    peak.type = 's2'
                else:
                    peak.type = 'coincidence'

        return event
