"""Definition of base classes for plugins

Here we define every one of the plugins/modules for pax.  This describes what
the interfaces are.  To add an input or a transform, you define one function
that does something with an event.  An input spits out event objects.  A
transform would modify the event object.

See format for more information on the event object.
"""
import logging
import os
from time import strftime

import numpy as np
import pax    # for version
from pax.datastructure import Event, ReconstructedPosition


class BasePlugin(object):
    # Processor.run() will ensure this gets set after it has shut down the plugin
    # If you ever shut down a plugin yourself, you need to set it too!!
    has_shut_down = False

    def __init__(self, config_values, processor):
        self.name = self.__class__.__name__
        self.processor = processor
        self.log = logging.getLogger(self.name)
        self.total_time_taken = 0   # Total time in msec spent in this plugin
        self.config = config_values
        self._pre_startup()
        y = self.startup()
        if y is not None:
            raise RuntimeError('Startup of %s returned a %s instead of None.' % (self.name, type(y)))

    def _pre_startup(self):
        pass

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
    # Set these to False if you don't want to check the input/output are actually pax events
    do_input_check = True
    do_output_check = True

    def _pre_startup(self):
        # Give the logger another name, we need self.log for the adapter
        self._log = self.log

    def process_event(self, event=None):
        if self.do_input_check:
            if not isinstance(event, Event):
                raise RuntimeError("%s received a %s instead of an Event" % (self.name, type(event)))
        # Setup the logging adapter which will prepend [Event: ...] to the logging messages
        self.log = EventLoggingAdapter(self._log, dict(event_number=event.event_number))
        if self.has_shut_down:
            raise RuntimeError("%s was asked to process an event, but it has already shut down!" % self.name)

        event = self._process_event(event)
        if self.do_output_check:
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

    def _pre_startup(self):
        # If no output name specified, create a default one.
        # We need to do this here, rather than in paxer, otherwise user couldn't specify output_name in config
        # (paxer would override it)
        if 'output_name' not in self.config:
            # Is there an input plugin? If so, try to use the input plugin's input name without extension.
            # This will give problems when both input and output have no extension (e.g. directories, databases),
            # but is very convenient otherwise.
            # Appending e.g. '_procesed' inevitably leads to '_processed_processed_...'
            ip = self.processor.input_plugin
            if ip is not None and 'input_name' in ip.config:
                self.config['output_name'] = os.path.splitext(os.path.basename(ip.config['input_name']))[0]
            else:
                # Deep fallback: timestamp-based name.
                self.config['output_name'] = 'output_pax%s_%s' % (pax.__version__, strftime('%y%m%d_%H%M%S'))
        if self.config['output_name'].endswith('/'):
            raise ValueError("Output names should not end with a slash. See issue #340.")
        ProcessPlugin._pre_startup(self)

    def write_event(self, event):
        """Do magic. Return None.
        """
        raise NotImplementedError

    def _process_event(self, event):
        result = self.write_event(event)
        if result is not None:
            raise RuntimeError("%s returned a %s instead of None" % (self.name, type(event)))
        return event


class PosRecPlugin(TransformPlugin):
    """Base plugin for position reconstruction
    Ensures all posrec plugins:
     - use the ReconstructedPosition.algorithm field in the same way (set to self.name)
     - act on the same set of peaks (all tpc peaks except lone-hits)
     - have the same behaviour when giving up (add a position with x = y = nan)
     - don't get passed peaks without top pmts active (we add the nan-position automatically)
     - have self.pmts and self.pmt_locations available in the same way
    """
    uses_only_top = True

    def _pre_startup(self):
        # List of integers of which PMTs to use, this algorithm uses the top pmt array to reconstruct
        if self.uses_only_top:
            self.pmts = np.array(self.config['channels_top'])
        else:
            self.pmts = np.array(self.config['channels_in_detector']['tpc'])

        # (x,y) Locations of these PMTs, stored as np.array([(x,y), (x,y), ...])
        self.pmt_locations = np.array([[self.config['pmts'][ch]['position']['x'],
                                        self.config['pmts'][ch]['position']['y']]
                                       for ch in self.pmts])

        TransformPlugin._pre_startup(self)

    def transform_event(self, event):
        for peak in event.get_peaks_by_type(detector='tpc'):
            # Do not act on lone hits
            if peak.type == 'lone_hit':
                continue

            # If there are no contributing top PMTs, don't even try:
            area_top = np.sum(peak.area_per_channel[self.pmts])
            if area_top == 0:
                pos_dict = None
            else:
                pos_dict = self.reconstruct_position(peak)

            # Parse the plugin's result
            if pos_dict is None:
                # The plugin gave up
                pos_dict = {}
            if isinstance(pos_dict, (list, tuple, np.ndarray)):
                # The plugin returned (x, y)
                pos_dict = dict(zip(('x', 'y'), pos_dict))

            # Add the algorithm field, then append the position to the peak
            pos_dict.update(dict(algorithm=self.name))
            peak.reconstructed_positions.append(ReconstructedPosition(**pos_dict))

        return event

    def reconstruct_position(self, peak):
        """Return a position {'x': ..., 'y': ...) or (x, y) for the peak or None (if you can't)."""
        raise NotImplementedError


class EventLoggingAdapter(logging.LoggerAdapter):
    """Prepends event number to log messages
    Adapted from https://docs.python.org/3.4/howto/logging-cookbook.html#context-info
    """
    def process(self, msg, kwargs):
        return '[Event %s] %s' % (self.extra['event_number'], msg), kwargs
