__author__ = 'tunnell'

import logging


class BasePlugin(object):

    def __init__(self, config_values):
        self.log = logging.getLogger(self.__class__.__name__)

        # Please do all config variable fetching in constructor to make
        # changing config easier.
        self.config = config_values

    def ProcessEvent(self):
        raise NotImplementedError()


class InputPlugin(BasePlugin):

    def __init__(self, config_values):
        BasePlugin.__init__(self, config_values)
        self.i = 0

    def GetEvents(self):
        """Get next event from the data source

        Raise a StopIteration when done
        """
        raise NotImplementedError()

    def ProcessEvent(self, event=None):
        raise RuntimeError('Input plugins cannot process data.')


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
