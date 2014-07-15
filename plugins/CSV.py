from pax import plugin

class WriteCSVPeakwise(plugin.OutputPlugin):
    def write_event(self,event):
        print(event)
