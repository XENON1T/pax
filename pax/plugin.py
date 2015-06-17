"""Definition of base classes for plugins

Here we define every one of the plugins/modules for pax.  This describes what
the interfaces are.  To add an input or a transform, you define one function
that does something with an event.  An input spits out event objects.  A
transform would modify the event object.

See format for more information on the event object.
"""
import logging

import time
from pax.datastructure import Event


class BasePlugin(object):

    def __init__(self, config_values, processor):
        self.name = self.__class__.__name__
        self.processor = processor
        self.log = logging.getLogger(self.name)
        self.total_time_taken = 0   # Total time in usec spent in this plugin
        # run() will ensure this gets set after it has shut down the plugin
        # If you ever shut down a plugin yourself, you need to set it too!!
        # TODO: this is clunky...
        self.has_shut_down = False

        # Please do all config variable fetching in constructor to make
        # changing config easier.
        # self.log.debug(config_values)
        self.config = config_values
        y = self.startup()
        if y is not None:
            raise RuntimeError('Startup of %s returned a %s instead of None.' % (
                self.name, type(y)))

    def __del__(self):
        if not self.has_shut_down:
            self.log.debug("Deleting %s, shutdown has NOT occurred yet!" % self.name)
            y = self.shutdown()
            if y is not None:
                raise RuntimeError('Shutdown of %s returned a %s instead of None.' % (self.name,
                                                                                      type(y)))
        #else:
        #    self.log.debug("Deleting %s, shutdown has already occurred" % self.name)

    @staticmethod
    def _timeit(method):
        """Decorator for measuring method speeds
        Should be wrapped about each plugin's main method
        """
        def timed(*args, **kw):
            self = args[0]
            ts = time.time()
            result = method(*args, **kw)
            dt = (time.time() - ts) * 1000
            self.total_time_taken += dt
            self.log.debug('Event took %2.2f ms' % dt)
            return result

        return timed

    def startup(self):
        self.log.debug("%s does not define a startup" % self.__class__.__name__)
        pass

    def process_event(self):
        raise NotImplementedError()

    def shutdown(self):
        pass


class InputPlugin(BasePlugin):

    """Base class for data inputs

    This class cannot be parallelized since events are read in a specific order
    """
    # The plugin should update this to the number of events which get_events will eventually return
    number_of_events = 0

    # TODO: how/where do we check if the input plugin has already shut down?

    def get_single_event(self, index):
        self.log.warning("Single event support not implemented for this input plugin... " +
                         "Iterating though events until we find event %s!" % index)
        for event in self.get_events():
            if event.event_number == index:
                return event

        raise RuntimeError("Event %d not found" % index)

    @BasePlugin._timeit
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

    @BasePlugin._timeit
    def process_event(self, event):
        if self.has_shut_down:
            raise RuntimeError("%s was asked to process an event, but it has already shut down!" % self.name)

        if event is None:
            raise RuntimeError(
                "%s transform received a 'None' event." % self.name)
        elif not isinstance(event, Event):
            raise RuntimeError("%s transform received wrongly typed event. %s" % (self.name,
                                                                                  event))
        #  The actual work is done in this line
        result = self.transform_event(event)

        if result is None:
            raise RuntimeError(
                "%s transform did not return event." % self.name)
        elif not isinstance(result, (dict, Event)):
            raise RuntimeError(
                "%s transform returned wrongly typed event." % self.name)

        return result


class OutputPlugin(BasePlugin):

    def write_event(self, event):
        raise NotImplementedError

    @BasePlugin._timeit
    def process_event(self, event):
        if self.has_shut_down:
            raise RuntimeError("%s was asked to process an event, but it has already shut down!" % self.name)
        self.write_event(event)
        return event
