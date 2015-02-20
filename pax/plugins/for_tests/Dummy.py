from pax import plugin, datastructure


class DummyInput(plugin.InputPlugin):

    def get_events(self):
        yield datastructure.Event.empty_event()


class DummyOutput(plugin.OutputPlugin):
    "Stores last event in self.last_event, useful for testing"

    def write_event(self, event):
        self.last_event = event


class DummyTransform(plugin.TransformPlugin):
    pass


class DummyTransform2(plugin.TransformPlugin):
    pass


class DummyTransform3(plugin.TransformPlugin):
    pass
