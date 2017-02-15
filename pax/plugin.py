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
from pax import dsputils
from pax.datastructure import Event, ReconstructedPosition, Peak, Hit


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
            # Appending e.g. '_processed' inevitably leads to '_processed_processed_...'
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


class ClusteringPlugin(TransformPlugin):
    """Base class for peak building / clustering plugins"""

    def transform_event(self, event):
        self.event = event
        new_peaks = []
        for peak in event.peaks:
            new_peaks += self.cluster_peak(peak)
        event.peaks = new_peaks

        # Update the event.all_hits field (for plotting), since new hits could have been created
        # Note we must separately get out the rejected hits, they are not in any peak...
        self.event.all_hits = np.concatenate([p.hits for p in self.event.peaks] +
                                             [self.event.all_hits[self.event.all_hits['is_rejected']]])

        # Restores order after shenanigans here
        # Important if someone uses (dict_)group_by from recarray tools later
        # As is done, for example in the hitfinder diagnostic plots... if you don't do this you get really strange
        # things there (missing hits were things were split, which makes you think there is a bug
        # in LocalMinimumClustering, but actually there isn't...)
        self.event.all_hits.sort(order='found_in_pulse')

        return event

    def cluster_peak(self, peak):
        """Takes peak and returns LIST of peaks"""
        raise NotImplementedError

    def build_peak(self, hits, detector, **kwargs):
        """Return a peak object made from hits. Compute a few basic properties which are needed during the clustering
        stages.
        Any kwargs will be passed to the peak constructor.
        """
        hits.sort(order='left_central')     # Hits must always be in sorted time order

        peak = Peak(detector=detector, hits=hits, **kwargs)

        peak.area_per_channel = dsputils.count_hits_per_channel(peak, self.config, weights=hits['area'])
        peak.n_contributing_channels = np.sum(peak.does_channel_contribute)

        if peak.n_contributing_channels == 0:
            raise RuntimeError("Every peak should have at least one contributing channel... what's going on?")

        if peak.n_contributing_channels == 1:
            peak.type = 'lone_hit'
            peak.lone_hit_channel = hits[0]['channel']

        peak.area = peak.area_per_channel.sum()
        peak.left = peak.hits[0]['left']
        peak.right = peak.hits['right'].max()

        return peak

    def split_peak(self, peak, split_points):
        """Yields new peaks split from peak at split_points = sample indices within peak
        Samples at the split points will fall to the right (so if we split [0, 5] on 2, you get [0, 1] and [2, 5]).
        Hits that straddle a split point are themselves split into two hits: peak.hits is updated.
        """
        # First, split hits that straddle the split points
        # Hits may have to be split several times; for each split point we modify the 'hits' list, splitting only
        # the hits we need.
        hits = peak.hits
        for x in split_points:
            x += peak.left   # Convert to index in event

            # Select hits that must be split: start before x and end after it.
            selection = (hits['left'] <= x) & (hits['right'] > x)
            hits_to_split = hits[selection]

            # new_hits will be a list of hit arrays, which we concatenate later to make the new 'hits' list
            # Start with the hits that don't have to be split: we definitely want to retain those!
            new_hits = [hits[True ^ selection]]

            for h in hits_to_split:
                pulse_i = h['found_in_pulse']
                pulse = self.event.pulses[pulse_i]

                # Get the pulse waveform in ADC counts above baseline (because it's what build_hits expect)
                baseline_to_subtract = self.config['digitizer_reference_baseline'] - pulse.baseline
                w = baseline_to_subtract - pulse.raw_data.astype(np.float64)

                # Use the hitfinder's build_hits to compute the properties of these hits
                # Damn this is ugly... but at least we don't have duplicate property computation code
                hits_buffer = np.zeros(2, dtype=Hit.get_dtype())
                adc_to_pe = dsputils.adc_to_pe(self.config, h['channel'])
                hit_bounds = np.array([[h['left'], x], [x+1, h['right']]], dtype=np.int64)
                hit_bounds -= pulse.left   # build_hits expects hit bounds relative to pulse start
                dsputils.build_hits(w,
                           hit_bounds=hit_bounds,
                           hits_buffer=hits_buffer,
                           adc_to_pe=adc_to_pe,
                           channel=h['channel'],
                           noise_sigma_pe=pulse.noise_sigma * adc_to_pe,
                           dt=self.config['sample_duration'],
                           start=pulse.left,
                           pulse_i=pulse_i,
                           saturation_threshold=self.config['digitizer_reference_baseline'] - pulse.baseline - 0.5,
                           central_bounds=hit_bounds)       # TODO: Recompute central_bounds in intelligent way

                new_hits.append(hits_buffer)

            # Now remake the hits list, then go on to the next peak.
            hits = np.concatenate(new_hits)

        # Next, split the peaks, sorting hits to the correct peak by their maximum index.
        # Iterate over left, right bounds of the new peaks
        boundaries = list(zip([0] + [y+1 for y in split_points], split_points + [float('inf')]))
        for l, r in boundaries:
            # Convert to index in event
            l += peak.left
            r += peak.left

            # Select hits which have their maximum within this peak bounds
            # The last new peak must also contain hits at the right bound (though this is unlikely to happen)
            hs = hits[(hits['index_of_maximum'] >= l) &
                      (hits['index_of_maximum'] <= r)]

            if not len(hs):
                print(l, r, hits['index_of_maximum'])
                raise RuntimeError("Attempt to create a peak without hits!")

            r = r if r < float('inf') else peak.right

            yield self.build_peak(hits=hs, detector=peak.detector, left=l, right=r)


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
