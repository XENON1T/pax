from pax import plugin


class XAMSClassification(plugin.TransformPlugin):
    def transform_event(self, event):

        for peak in event.peaks:

            # Work only on unknown peaks - not noise, lone_pulse, and big peaks (which are already s1 or s2)
            if peak.type != 'unknown':
                continue

            if (peak.right - peak.left) < 500:
                peak.type = 's1'
                continue

            elif (peak.right - peak.left) > 500 and peak.area > 8:
                peak.type = 's2'
                continue

        return event
