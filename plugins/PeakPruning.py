from pax import plugin, units

class PeakPruner(plugin.TransformPlugin):
    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

    def transform_event(self, event):
        for peak_index,p in enumerate(event['peaks']):
            if not 'rejected' in p: 
                p['rejected'] = False
                p['rejection_reason'] = None
                p['rejected_by'] = None
            if p['rejected']:
                continue
            (decision, reason) = self.decide_peak(peak_index, event)
            if decision == 'reject':
                event['peaks'][i]['rejected'] = True
                p['rejection_reason'] = reason
                p['rejected_by'] = self
        return event
        
    def decide_peak(self, peak_index, event):
        raise NotImplementedError("This peak decider forgot to implement decide_peak...")
        
class PruneWideS1s(PeakPruner):
    def __init__(self, config):
        PeakPruner.__init__(self, config)
        
    def decide_peak(self, peak_index, event):
        peak = event['peaks'][peak_index]
        fwqm = peak['top_and_bottom']['fwqm']
        treshold = 0.5 * units.us
        if fwqm > treshold:
            return ('rejected', 'S1 FWQM is %s us, higher than maximum %s us.' % (fwqm, treshold))
        else: return ('accepted',None)
        
class PruneS1sInS2Tails(PeakPruner):
    def __init__(self, config):
        PeakPruner.__init__(self, config)
        
    def decide_peak(self, peak_index, event):
        peak = event['peaks'][peak_index]
        fwqm = peak['top_and_bottom']['fwqm']
        treshold = 0.5 * units.us
        if fwqm > treshold:
            return ('rejected', 'S1 FWQM is %s us, higher than maximum %s us.' % (fwqm, treshold))
        else: return ('accepted',None)