from pax import plugin


class AdHocClassification(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:

            width = peak.hit_time_std

            # Work only on unknown peaks - not noise and lone_hit
            if peak.type != 'unknown':
                continue

            if peak.area > 30:

                if width < 120:
                    peak.type = 's1'
                elif width > 200:
                    peak.type = 's2'

            else:
                # For smaller peaks, hit_time_std is a bit less.
                # Also have to worry about single electrons.

                if width < 80:
                    peak.type = 's1'
                elif width > 140 and peak.area > 8:
                    peak.type = 's2'

        return event
