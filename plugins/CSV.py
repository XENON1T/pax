from pax import plugin

import csv

class WriteCSVPeakwise(plugin.OutputPlugin):

    def __init__(self,config):
        plugin.OutputPlugin.__init__(self,config)
        self.counter = 0

    def write_event(self,event):
        peaks_flattened = [self.flatten_to_csvline(p) for p in event['peaks']]
        peaks_flattened = sorted(peaks_flattened, key= lambda x: x['left'])
        for p in peaks_flattened:
            data = {
                'event' :       self.counter,    #TODO: get from mongo/xed/whatever
                'left'  :       p['left'],
                'right' :       p['right'],
                'area'  :       p['top_and_bottom|area'],
                'type'  :       p['peak_type'],
                'rejected'  :   p['rejected'],
                'rejected_by'  :   p['rejected_by'],
                'rejection_reason'  :   p['rejection_reason'],
            }
            if not hasattr(self, 'csv'):
                self.output = open('output.csv', 'w')
                self.headers = ['event', 'type', 'left', 'right', 'area', 'rejected', 'rejected_by', 'rejection_reason'] #Grmpfh, needed for order
                self.csv = csv.DictWriter(self.output, self.headers, lineterminator='\n')
                self.csv.writeheader()
            self.csv.writerow(data)
        self.counter += 1
            
    def flatten_to_csvline(self, datastructure, prefix=''):
        "Changes a nested dictionary / array / tuple  thing into a single csv line. Returns results as dictionary {header: value, header:value, ... }"
        results = {}
        if type(datastructure) in (type({}),type([]),type((0,0))):
            if prefix != '': prefix += '|'
            if type(datastructure) == type({}):
                temp = [self.flatten_to_csvline(value, prefix=prefix+str(key)) for key, value in datastructure.items()] 
            else:
                temp = [self.flatten_to_csvline(value, prefix=prefix+str(key)) for key, value in enumerate(datastructure)]
            #Merge results into one dict
            for subdict in temp:
                results.update(subdict)
        else:
            results[prefix] = str(datastructure)
        return results