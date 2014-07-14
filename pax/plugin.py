__author__ = 'tunnell'

import logging
import time


class BasePlugin(object):

    def __init__(self, config_values):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)

        # Please do all config variable fetching in constructor to make
        # changing config easier.
        self.config = config_values

    def ProcessEvent(self):
        raise NotImplementedError()


class InputPlugin(BasePlugin):
    """Base class for data inputs

    This class cannot be parallelized since events are read in a specific order
    """
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
        t0 = time.time()
        event = self.TransformEvent(event)
        dt = (time.time() - t0)*10**3  # milliseconds
        self.log.debug('Class %s took %0.1f milliseconds' % (self.name, dt))
        return event


class OutputPlugin(BasePlugin):

    def WriteEvent(self, event):
        raise NotImplementedError

    def ProcessEvent(self, event):
        self.WriteEvent(event)
        return event
