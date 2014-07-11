from pax import plugin


class DumbExampleInput(plugin.InputPlugin):
    def __init__(self):
        plugin.InputPlugin.__init__(self)
        self.mycounter = 0

    def GetNextEvent(self):
        self.mycounter += 1
        if self.mycounter > 10:
            raise StopIteration
        return {}



class DumbExampleTransform(plugin.TransformPlugin):
    def TransformEvent(self, event):
        event['ya'] = 1
        return event



class DumbExampleOutput(plugin.OutputPlugin):
    def WriteEvent(self, event):
        self.log.fatal("writing event to screen %s" % str(event))


