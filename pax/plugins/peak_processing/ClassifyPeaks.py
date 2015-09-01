from pax import plugin


class AdHocClassification(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:

            width = peak.range_90p_area

            # Work only on unknown peaks - not noise and lone_hit
            if peak.type != 'unknown':
                continue

            if width < 125:
                peak.type = 's1'
            elif width > 200:
                if peak.area > 5:
                    peak.type = 's2'
                else:
                    peak.type = 'coincidence'

        return event
