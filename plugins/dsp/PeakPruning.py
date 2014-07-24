from pax import plugin, units
import numpy as np

# TODO: this part is not obvious, you need a paragraph on what pruning is
# TODO: does order of prunning matter?


# decision: none: accept, string: reject, string specifies reason


def is_s2(peak):  # put in plugin base class?
    if 'rejected' in peak and peak['rejected']: return False
    return peak['peak_type'] in ('large_s2', 'small_s2')


class PeakPruner(plugin.TransformPlugin):

    def transform_event(self, event):
        for peak_index, p in enumerate(event['peaks']):
            # If this is the first peak pruner, we have to set up some values
            if not 'rejected' in p:
                p['rejected'] = False
                p['rejection_reason'] = None
                p['rejected_by'] = None

            # If peak has been rejected earlier, we don't have to test it
            # In the future we may want to disable this to test how the
            # prunings depend on each other
            if p['rejected']:
                continue
            # Child class has to define decide_peak
            decision = self.decide_peak(p, event, peak_index)
            # None means accept the peak. Anything else is a rejection reason.
            if decision != None:
                p['rejected'] = True
                p['rejection_reason'] = decision
                p['rejected_by'] = str(self.__class__.__name__)
        return event

    def decide_peak(self, peak, event, peak_index):
        raise NotImplementedError("This peak pruner forgot to implement decide_peak...")



class PruneS2sInS2Tails(PeakPruner):

    def decide_peak(self, peak, event, peak_index):
        if peak['peak_type'] != 'small_s2':
            return None
        if not 'stop_looking_for_s2s_after' in event:
            # Determine where to stop looking for S2s
            # Stop if there is an earlier S2 whose amplitude exceeds a treshold
            treshold = 624.151  # S2 amplitude after which no more s2s are looked for
            larges2boundaries = [p['left'] for p in event['peaks'] if is_s2(p) and p['top_and_bottom']['height'] > treshold]
            if larges2boundaries == []:
                # No large S2s in this waveform - S2 always ok
                event['stop_looking_for_s2s_after'] = float('inf')
                return None
            event['stop_looking_for_s2s_after'] = min(larges2boundaries)
        if peak['left'] > event['stop_looking_for_s2s_after']:
            return 'S2 starts at %s, which is beyond %s, the starting position of a "large" S2.' % (peak['left'], event['stop_looking_for_s2s_after'])
        return None