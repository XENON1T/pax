#from pax import plugin

import csv

class Peakwise(plugin.OutputPlugin):
    def __init__(self,config):
        plugin.OutputPlugin.__init__(self,config)

    def WriteEvent(self,event):
        peaks_flattened = [self.flatten_to_csvline(p) for p in peaks]
        
        for p in peaks_flattened:
            if not hasattr(self, csv):
                self.output = open('output.csv', 'w')
                self.headers = event['peaks']
                self.csv = csv.DictWriter(output, p.keys())
                self.csv.writeheader()
            self.csv.writerow(p)
            
    def flatten_to_csvline(self, datastructure, prefix=''):
        "Changes a nested dictionary / array / tuple  thing into a single csv line. Returns results as dictionary {header: value, header:value, ... }"
        results = {}
        if type(datastructure) in (types.DictType,types.ListType, types.TupleType):
            if prefix != '': prefix += '|'
            if type(datastructure) == types.DictType:
                temp = [self.flatten_to_csvline(value, prefix=prefix+str(key)) for key, value in datastructure.items()] 
            else:
                temp = [self.flatten_to_csvline(value, prefix=prefix+str(key)) for key, value in enumerate(datastructure)]
            #Merge results into one dict
            for subdict in temp:
                results.update(subdict)
        else:
            results[prefix] = str(datastructure)
        return results
            

