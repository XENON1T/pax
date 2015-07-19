from pax import plugin, datastructure


class DummyInput(plugin.InputPlugin):
    """Yield empty events with consecutive event numbers"""
    events_yielded = 0

    def get_events(self):
        while True:
            event = datastructure.Event.empty_event()
            event.event_number = self.events_yielded
            self.events_yielded += 1
            yield event


class DummyOutput(plugin.OutputPlugin):
    "Counts events 'written', and stores last event in self.last_event. Useful for testing"
    events_written = 0

    def write_event(self, event):
        self.events_written += 1
        self.last_event = event


class DummyTransform(plugin.TransformPlugin):
    pass


class DummyTransform2(plugin.TransformPlugin):
    pass


class DummyTransform3(plugin.TransformPlugin):
    pass


class SelectEvenEvents(plugin.SelectionPlugin):
    """Select event all events with an even event number"""

    def test_event(self, event):
        return event.event_number % 2 == 0
