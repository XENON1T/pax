from pax import plugin

class MyDumbExample(plugin.InputPlugin, plugin.TransformPlugin, plugin.OutputPlugin):
    def GetNextEvent(self):
        return None

