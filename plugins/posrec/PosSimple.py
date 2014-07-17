"""Very simple position reconstruction for s2"""
from pax import plugin

__author__ = 'coderre'

class PosRecWeightedSum(plugin.TransformPlugin):
    """Class to reconstruct s2 x,y positions using the weighted sum of PMTs in the top array. Positions stored in peak['rec']['PosSimple']"""

    def __init__(self,config):
        plugin.TransformPlugin.__init__(self,config)
        self.topArrayMap = config['topArrayMap']
        
    def transform_event(self,event):
        for peak in event['peaks']:
            if 'rec' not in peak.keys():
                peak['rec']={}
            if peak['peak_type'] == 'large_s2' or peak['peak_type'] == 'small_s2':
                peak['rec']['PosSimple']={}
                peak_x = peak_y = peak_E = 0.
                for i in range(0,len(self.topArrayMap)):                    
                    if peak['hit_list'][i]!=0:
                        peak_x+=peak['hit_list'][i]*self.topArrayMap[i]['x']
                        peak_y+=peak['hit_list'][i]*self.topArrayMap[i]['y']
                        peak_E+=peak['hit_list'][i]
                if peak_E !=0:
                    peak_x/=peak_E
                    peak_y/=peak_E
                else:
                    peak_x=peak_y=-500.            #algorithm failed
                peak['rec']['PosSimple']={}
                peak['rec']['PosSimple']={'x':peak_x,'y':peak_y}
        return event
