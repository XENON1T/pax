from pax import datastructure
from pax import plugin

class ConvertToEventClass(plugin.TransformPlugin):
    def transform_event(self, event):
        from pprint import pprint
        #pprint.pprint(event['peaks'])
        #print(event.keys)
        event = datastructure.Event(event)
        pprint(event.S2s())
        return event
