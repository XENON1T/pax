from pax import plugin, datastructure


class DummyInput(plugin.InputPlugin):

    def get_events(self):
        yield datastructure.Event.empty_event()


class DummyOutput(plugin.OutputPlugin):
    pass


class DummyTransform(plugin.TransformPlugin):
    pass


class DummyTransform2(plugin.TransformPlugin):
    pass


class DummyTransform3(plugin.TransformPlugin):
    pass
