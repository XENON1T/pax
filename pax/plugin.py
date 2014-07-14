__author__ = 'tunnell'

import logging
import time


def timeit(method):
    """Decorator for measuring class method speeds
    """

    def timed(*args, **kw):
        self = args[0]
        ts = time.time()
        result = method(*args, **kw)
        dt = (time.time() - ts) * 100
        self.log.debug('Event took %2.2f ms' % dt)
        return result

    return timed


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

    @timeit
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

    @timeit
    def ProcessEvent(self, event):
        return self.TransformEvent(event)


class OutputPlugin(BasePlugin):
    def WriteEvent(self, event):
        raise NotImplementedError

    @timeit
    def ProcessEvent(self, event):
        self.WriteEvent(event)
        return event
