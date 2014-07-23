from pax import datastructure
from pax import plugin

class ConvertToEventClass(plugin.TransformPlugin):
    def transform_event(self, event):
        import pprint
        pprint.pprint(event['peaks'])
        event = datastructure.Event(event)

        return event