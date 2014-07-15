from pax import plugin, units

# decision: none: accept, string: reject, string specifies reason


def is_s2(peak):
    return peak['peak_type'] in ('large_s2', 'small_s2')


class PeakPruner(plugin.TransformPlugin):

    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

    def transform_event(self, event):
        for peak_index, p in enumerate(event['peaks']):
            if not 'rejected' in p:
                p['rejected'] = False
                p['rejection_reason'] = None
                p['rejected_by'] = None
            if p['rejected']:
                continue
            decision = self.decide_peak(p, event, peak_index)
            if decision != None:
                p['rejected'] = True
                p['rejection_reason'] = decision
                p['rejected_by'] = self
        return event

    def decide_peak(self, peak, event, peak_index):
        raise NotImplementedError("This peak decider forgot to implement decide_peak...")


class PruneWideS1s(PeakPruner):

    def __init__(self, config):
        PeakPruner.__init__(self, config)

    def decide_peak(self, peak, event, peak_index):
        if peak['peak_type'] != 's1':
            return
        fwqm = peak['top_and_bottom']['fwqm']
        treshold = 0.5 * units.us
        if fwqm > treshold:
            return 'S1 FWQM is %s us, higher than maximum %s us.' % (fwqm / units.us, treshold / units.us)
        return

        
class PruneWideShallowS2s(PeakPruner):
    def __init__(self, config):
        PeakPruner.__init__(self, config)
        
    def decide_peak(self, peak, event, peak_index):
        if peak['peak_type'] != 'small_s2': return
        treshold = 0.1 *units.mV/units.ns #S2 amplitude after which no more s1s are looked for
        
        if peak['left'] > self.earliestboundary:
            return 'S1 starts at %s, which is beyond %s, the starting position of a "large" S2.' % (peak['left'], self.earliestboundary)
        return
        
class PruneS1sWithNearbyNegativeExcursions(PeakPruner):
    def __init__(self, config):
        PeakPruner.__init__(self, config)
        
    def decide_peak(self, p, event, peak_index):
        if p['peak_type'] != 's1': return
        data = event['sum_waveforms']['top_and_bottom']
        negex = p['lowest_nearby_value'] = min(data[
            max(0,p['left']-500) :
            min(len(data),p['right'] + 101)
        ])  #Window used by s1 filter: todo: don't hardcode    
        maxval =  p['top_and_bottom']['height']
        factor = 3
        if negex<0 and factor * abs(negex) >  maxval:
            return 'Nearby negative excursion of %s, height (%s) not at least %s x as large.' % (negex, maxval, factor)
        return
        
class PruneS1sInS2Tails(PeakPruner):

    def __init__(self, config):
        PeakPruner.__init__(self, config)

    def decide_peak(self, peak, event, peak_index):
        if peak['peak_type'] != 's1':
            return
        treshold = 3.12255  # S2 amplitude after which no more s1s are looked for
        if not hasattr(self, 'earliestboundary'):
            s2boundaries = [p['left'] for p in event['peaks'] if is_s2(p) and p['top_and_bottom']['height'] > treshold]
            if s2boundaries == []:
                self.earliestboundary = float('inf')
            else:
                self.earliestboundary = min(s2boundaries)
        if peak['left'] > self.earliestboundary:
            return 'S1 starts at %s, which is beyond %s, the starting position of a "large" S2.' % (peak['left'], self.earliestboundary)
        return

