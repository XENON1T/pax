"""Very simple position reconstruction for s2"""
from pax import plugin

import numpy as np

class PosRecWeightedSum(plugin.TransformPlugin):

    """Class to reconstruct s2 x,y positions using the weighted sum of PMTs in the top array. Positions stored in
    peak['rec']['PosSimple']"""

    #def startup(self):
    #    self.top_array_map = self.config['topArrayMap']
    #    self.num_channels = self.config['num_pmts']

    def transform_event(self, event):
        assert print != True
        print('hi')

        #self.log.debug("PMT waveforms: %s" % str(event.pmt_waveforms))
        #for peak in event.peaks:
        #    print('test')
        #    self.log.debug("Left %d, Right: %d" % (event.left, event.right))
        #    hits = event.pmt_waveforms[..., peak.left:peak.right].sum(axis=1)


        return event
