from pax import plugin

class WriteCSVPeakwise(plugin.OutputPlugin):
    def WriteEvent(self,event):
        print(event)
