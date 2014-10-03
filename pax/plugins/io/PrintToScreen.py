from pax import plugin


class PrintToScreen(plugin.OutputPlugin):

    def write_event(self, event):
        self.log.info(event.to_json())
