from pax import plugin


class AdHocClassification(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:

            # Work only on unknown peaks - not noise and lone_hit
            if peak.type != 'unknown':
                continue

            if peak.area > 30:

                if peak.hit_time_std < 80:
                    peak.type = 's1'
                elif peak.hit_time_std > 200:
                    peak.type = 's2'

            else:

                # For smaller peaks, hit_time_std is a bit less.
                # Also have to worry about single electrons.

                if peak.hit_time_std < 60:
                    peak.type = 's1'
                elif peak.hit_time_std > 120:
                    peak.type = 's2'

        return event
