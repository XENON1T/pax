from pax import plugin

__author__ = 'tunnell'


class PrintToScreen(plugin.OutputPlugin):

    def WriteEvent(self, event):
        self.log.fatal("writing event to screen %s" % str(event))
