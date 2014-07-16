from pax import plugin

import csv


class WriteCSVPeakwise(plugin.OutputPlugin):

    def __init__(self, config):
        plugin.OutputPlugin.__init__(self, config)
        self.counter = 0

    def write_event(self, event):
        for p in event['peaks']:
            data = {
                'event':      self.counter,  # TODO: get from mongo/xed/whatever
                'left':       p['left'],
                'right':      p['right'],
                'area':       p['top_and_bottom']['area'],
                'type':       p['peak_type'],
                'rejected':   p['rejected'],
                'rejected_by':p['rejected_by'],
                'rejection_reason':   p['rejection_reason'],
            }
            if not hasattr(self, 'csv'):
                self.output = open('output.csv', 'w')
                self.headers = ['event', 'type', 'left', 'right', 'area', 'rejected', 'rejected_by', 'rejection_reason']  # Grmpfh, needed for order
                self.csv = csv.DictWriter(self.output, self.headers, lineterminator='\n')
                self.csv.writeheader()
            self.csv.writerow(data)
        self.counter += 1