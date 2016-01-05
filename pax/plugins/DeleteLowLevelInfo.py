from pax import plugin
import numpy as np


class DeleteLowLevelInfo(plugin.TransformPlugin):
    """This deletes all low-level info from the datastructure, to make the output file smaller
    This is what gets removed:
      * hits for all but the main s1
      * pulses for all but the main s1
      * sum waveforms (but not the peak sum waveforms... maybe in the future)
    """

    def transform_event(self, event):
        pulses_to_keep = []
        for p in event.peaks:
            if p.type == 's1':
                pulses_to_keep.extend(p.hits['found_in_pulse'].tolist())
            else:
                p.hits = p.hits[:0]   # Set hits to an empty array
        event.all_hits = event.all_hits[0:]
        event.pulses = [p for i, p in enumerate(event.pulses) if i in np.unique(pulses_to_keep)]
        event.sum_waveforms = []
        return event
