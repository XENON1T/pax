from pax import plugin


class DumpHits(plugin.TransformPlugin):

    def transform_event(self, event):
        main_s1 = event.main_s1
        for p in event.peaks:
            if p == main_s1:
                continue
            p.hits = p.hits[:0]   # Set hits to an empty array
        return event
