from pax import plugin, datastructure

class DummyInput(plugin.InputPlugin):

    def get_events(self):
        yield datastructure.Event()


class DummyOutput(plugin.OutputPlugin):
    pass

class DummyTransform(plugin.TransformPlugin):
    pass