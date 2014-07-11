__author__ = 'tunnell'

import logging

class BasePlugin():
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def ProcessEvent(self):
        raise NotImplementedError()

class InputPlugin(BasePlugin):
    def __init__(self):
        BasePlugin.__init__(self)
        self.i = 0

    def GetNextEvent(self):
        """Get next event from the data source

        Raise a StopIteration when done
        """
        raise NotImplementedError()

    def ProcessEvent(self, event=None):
        return self.GetNextEvent()

class TransformPlugin(BasePlugin):
    def TransformEvent(self, event):
        raise NotImplementedError

    def ProcessEvent(self, event):
        return self.TransformEvent(event)

class OutputPlugin(BasePlugin):
    def WriteEvent(self, event):
        raise NotImplementedError

    def ProcessEvent(self, event):
        self.WriteEvent(event)
        return event
