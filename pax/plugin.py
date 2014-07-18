"""Definition of base classes for plugins

Here we define every one of the plugins/modules for pax.  This describes what
the interfaces are.  To add an input or a transform, you define one function
that does something with an event.  An input spits out event objects.  A
transform would modify the event object.

See format for more information on the event object.
"""
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

    def process_event(self):
        raise NotImplementedError()


class InputPlugin(BasePlugin):
    """Base class for data inputs

    This class cannot be parallelized since events are read in a specific order
    """

    def __init__(self, config_values):
        BasePlugin.__init__(self, config_values)
        self.i = 0

    @timeit
    def get_events(self):
        """Get next event from the data source

        Raise a StopIteration when done
        """
        raise NotImplementedError()

    def process_event(self, event=None):
        raise RuntimeError('Input plugins cannot process data.')


class TransformPlugin(BasePlugin):

    def transform_event(self, event):
        raise NotImplementedError

    @timeit
    def process_event(self, event):
        return self.transform_event(event)


class OutputPlugin(BasePlugin):

    def write_event(self, event):
        raise NotImplementedError

    @timeit
    def process_event(self, event):
        self.write_event(event)
        return event
