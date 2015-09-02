from pax import plugin, units


class AdHocClassification(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:

            # Don't work on noise and lone_hit
            if peak.type in ('unknown', 'lone_hit'):
                continue

            if peak.range_90p_area < 150 * units.ns:
                peak.type = 's1'

            elif peak.range_90p_area > 200 * units.ns:
                if peak.area > 5:
                    peak.type = 's2'
                else:
                    peak.type = 'coincidence'

        return event
