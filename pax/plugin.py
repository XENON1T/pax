"""Definition of base classes for plugins

Here we define every one of the plugins/modules for pax.  This describes what
the interfaces are.  To add an input or a transform, you define one function
that does something with an event.  An input spits out event objects.  A
transform would modify the event object.

See format for more information on the event object.
"""
import logging

from pax.datastructure import Event
from pax import exceptions


class BasePlugin(object):

    def __init__(self, config_values, processor):
        self.name = self.__class__.__name__
        self.processor = processor
        self.log = logging.getLogger(self.name)
        self.total_time_taken = 0   # Total time in msec spent in this plugin

        # run() will ensure this gets set after it has shut down the plugin
        # If you ever shut down a plugin yourself, you need to set it too!!
        # TODO: this is clunky...
        self.has_shut_down = False

        # Please do all config variable fetching in constructor to make changing config easier.
        self.config = config_values
        y = self.startup()
        if y is not None:
            raise RuntimeError('Startup of %s returned a %s instead of None.' % (self.name, type(y)))

    def __del__(self):
        if not self.has_shut_down:
            self.log.debug("Deleting %s, shutdown has NOT occurred yet!" % self.name)
            y = self.shutdown()
            if y is not None:
                raise RuntimeError('Shutdown of %s returned a %s instead of None.' % (self.name, type(y)))
        else:
            self.log.debug("Deleting %s, shutdown has already occurred" % self.name)

    def startup(self):
        self.log.debug("%s does not define a startup" % self.__class__.__name__)
        pass

    def shutdown(self):
        pass


class InputPlugin(BasePlugin):
    """Base class for data inputs

    This class cannot be parallelized since events are read in a specific order
    """
    # The plugin should update this to the number of events which get_events
    # will eventually return
    number_of_events = 0

    # TODO: we never check if the input plugin has already shut down...

    def get_single_event(self, index):
        self.log.warning("Single event support not implemented for this input plugin... " +
                         "Iterating though events until we find event %s!" % index)
        for event in self.get_events():
            if event.event_number == index:
                return event

        raise RuntimeError("Event %d not found" % index)

    def get_events(self):
        """Iterate over all events in the data source"""
        raise NotImplementedError


class ProcessPlugin(BasePlugin):
    """Plugin that can process events"""

    def process_event(self, event=None):
        if not isinstance(event, Event):
            raise RuntimeError("%s received a %s instead of an Event" % (self.name, type(event)))
        if self.has_shut_down:
            raise RuntimeError("%s was asked to process an event, but it has already shut down!" % self.name)
        event = self._process_event(event)
        if not isinstance(event, Event):
            raise RuntimeError("%s returned a %s instead of an event." % (self.name, type(event)))
        return event

    def _process_event(self, event):
        raise NotImplementedError


class TransformPlugin(ProcessPlugin):

    def transform_event(self, event):
        """Do your magic. Return event"""
        raise NotImplementedError

    def _process_event(self, event):
        return self.transform_event(event)


class OutputPlugin(ProcessPlugin):

    def write_event(self, event):
        """Do magic. Return None.
        """
        raise NotImplementedError

    def _process_event(self, event):
        result = self.write_event(event)
        if result is not None:
            raise RuntimeError("%s returned a %s instead of None" % (self.name, type(event)))
        return event


class SelectionPlugin(ProcessPlugin):
    def __init__(self, *args, **kwargs):
        self.events_selected = 0
        self.events_seen = 0
        super().__init__(*args, **kwargs)

    def test_event(self, event):
        """Do magic. Return True (event passes) or False (event fails, continue immediately to next event)"""
        raise NotImplementedError

    def _process_event(self, event):
        self.events_seen += 1
        result = self.test_event(event)
        if not isinstance(result, bool):
            raise RuntimeError("%s returned a %s instead of True or False" % (self.name, type(event)))
        if result:
            self.events_selected += 1
            return event
        else:
            raise exceptions.SkipEvent

    @property
    def fraction_accepted(self):
        return self.events_selected / self.events_seen
