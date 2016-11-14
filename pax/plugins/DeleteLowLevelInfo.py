from pax import plugin
import numpy as np


class DeleteLowLevelInfo(plugin.TransformPlugin):
    """This deletes low-level info from the datastructure, to make the output file smaller.
    By default, this is what gets removed:
      * hits for all but the main s1
      * pulses for all but the main s1
      * sum waveforms (but not the peak sum waveforms stored with each peak)
    """

    def transform_event(self, event):

        # For high energy events, zero the data in expensive fields, except for the 5 largest S1s and S2s in the TPC
        if event.n_pulses > self.config.get('shrink_data_threshold', float('inf')):
            largest_indices = [event.peaks.index(x) for x in (event.s1s()[:5] + event.s2s()[:5])]
            for i, p in enumerate(event.peaks):
                if i in largest_indices:
                    continue
                p.sum_waveform *= 0
                p.sum_waveform_top *= 0
                p.area_per_channel *= 0
                p.hits_per_channel *= 0
                p.n_saturated_per_channel *= 0

        if self.config.get('delete_sum_waveforms', True):
            event.sum_waveforms = []

        delopt = self.config.get('delete_hits_and_pulses', 'not_for_s1s')

        if not delopt or delopt == 'none':
            pass

        elif delopt == 'not_for_s1s':
            pulses_to_keep = []
            for p in event.peaks:
                if p.type == 's1':
                    pulses_to_keep.extend(p.hits['found_in_pulse'].tolist())
                else:
                    p.hits = p.hits[:0]   # Set hits to an empty array
            pulses_to_keep = np.unique(pulses_to_keep)
            event.all_hits = event.all_hits[:0]
            event.pulses = [p for i, p in enumerate(event.pulses) if i in pulses_to_keep]

        elif delopt == 'all':
            event.pulses = []
            for p in event.peaks:
                p.hits = p.hits[:0]

        else:
            raise ValueError("Illegal delete_hits_and_pulses value %s" % delopt)

        return event
