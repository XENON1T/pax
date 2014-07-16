from pax import plugin

class PosRecWeightedSum(plugin.TransformPlugin):
    def __init__(self,config):
        plugin.TransformPlugin.__init__(self,config)
        self.topArrayMap = config['topArrayMap']
        
    def transform_event(self,event):
        for peak in event['peaks']:
            peak['rec']={}
            if peak['peak_type'] == 'large_s2' or peak['peak_type'] == 'small_s2':
                peak['rec']['PosSimple']={}
                peak_x = 0.
                peak_y = 0.
                peak_E = 0.                
                for i in range(0,len(self.topArrayMap)):                    
                    if peak['hit_list'][i]!=0:
                        peak_x+=peak['hit_list'][i]*self.topArrayMap[i]['x']
                        peak_y+=peak['hit_list'][i]*self.topArrayMap[i]['y']
                        peak_E+=peak['hit_list'][i]
                if peak_E !=0:
                    peak_x/=peak_E
                    peak_y/=peak_E
                peak['rec']['PosSimple']={}
                peak['rec']['PosSimple']={'x':peak_x,'y':peak_y}
        return event
