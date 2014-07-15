from pax import plugin, units

class PeakCutter(plugin.TransformPlugin):
    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

    def TransformEvent(self, event):
        for i,p in enumerate(event['peaks']):
            if not 'rejected' in p: 
                p['rejected'] = False
                p['rejection_reason'] = None
                p['rejected_by'] = None
            if p['rejected']:
                continue
            (decision, reason) = self.decide_peak(peakIndex, event):
            if decision == 'reject':
                event['peaks'][i]['rejected'] = True
                p['rejection_reason'] = reason
                p['rejected_by'] = self
        return event
        
    def decide_peak(self, peakIndex, event):
        raise NotImplementedError()
        
class S1FWQMTest(PeakCutter):
    def __init__(self, config):
        PeakCutter.__init__(self, config)
        
    def decide_peak(self, peakIndex, event):
        peak = event['peaks'][peakIndex]
        fwqm = peak['summed']['fwqm']
        treshold = 0.5 * units.us
        if fwqm < treshold:
            return ('rejected', 'S1 FWQM is %s us, lower than required %s us.' % (fwqm, treshold))
        else: return ('accepted',)