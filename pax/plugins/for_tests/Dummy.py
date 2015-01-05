from pax import plugin, datastructure, utils


class DummyInput(plugin.InputPlugin):

    def get_events(self):
        yield utils.empty_event()


class DummyOutput(plugin.OutputPlugin):
    pass


class DummyTransform(plugin.TransformPlugin):
    pass


class DummyTransform2(plugin.TransformPlugin):
    pass


class DummyTransform3(plugin.TransformPlugin):
    pass
