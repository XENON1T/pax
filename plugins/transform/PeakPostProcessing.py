"""Post processing for peaks. Should include any plugins run directly after peak finding to prep information needed for more complex transforms"""
from pax import plugin
import numpy

class MakeHitList(plugin.TransformPlugin):
    """ Class to make a hit list for each s1 and s2 peak as well as the multiplicity. Hit list stored in list with length equal to number of PMTs"""

    def __init__(self,config):
        plugin.TransformPlugin.__init__(self,config)
        print(config)
        self.num_channels = config['num_pmts']

    def transform_event(self,event):
        for peak in event['peaks']:
            hitList = numpy.zeros(self.num_channels)
            nHits = 0
            for i in event['channel_waveforms'].keys():                
                waveform = event['channel_waveforms'][i]
                hitList[i]+=sum(waveform[peak['left']:peak['right']])
                if hitList[i]!= 0:
                    nHits+=1
            peak['n_hits']=nHits
            peak['hit_list']=hitList
        return event
