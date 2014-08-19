import csv

from pax import plugin

class WritePeaksCSV(plugin.OutputPlugin):

    def write_event(self, event):
        event.peaks.sort(key=lambda p: p.index_of_maximum)
        for p in event.peaks:
            data = (
                ('event', event.event_number),
                ('type',  p.type),
                ('max',   p.index_of_maximum),
                ('left',  p.left),
                ('right', p.right),
                ('area',  p.area),
            )
            if not hasattr(self, 'csv'):
                # If I do this in startup, I have to copy-paste the fieldnames
                self.output = open('output.csv', 'w')
                self.headers = [q[0] for q in data]
                self.csv = csv.DictWriter(self.output, self.headers, lineterminator='\n')
                self.csv.writeheader()
            self.csv.writerow({a[0]: a[1] for a in data})

    def shutdown(self):
        self.output.close()

# class WriteEventsToCSV(plugin.OutputPlugin):
#
#     def startup(self):
#         self.counter = -1
#         self.output = open('output.csv', 'w')
#         self.headers = ['event', 'largest_s2_area', 'largest_s2_left', 'largest_s2_right']  # Grmpfh, needed for order
#         self.csv = csv.DictWriter(self.output, self.headers, lineterminator='\n')
#         self.csv.writeheader()
#
#     def write_event(self, event):
#         self.counter += 1
#
#         # Find largest s2...
#         s2areas = [p['top_and_bottom']['area'] for p in event['peaks'] if is_s2(p)]
#         if s2areas == []:
#             # No S2s in this waveform - skip event
#             return
#         # DANGER ugly code ahead...
#         s2maxarea = max(s2areas)
#         for i, p in enumerate(event['peaks']):
#             if p['top_and_bottom']['area'] == s2maxarea:
#                 largests2 = i
#
#         self.csv.writerow({
#             'event': self.counter,
#             'largest_s2_area': event['peaks'][largests2]['top_and_bottom']['area'],
#             'largest_s2_left': event['peaks'][largests2]['left'],
#             'largest_s2_right': event['peaks'][largests2]['right']
#         })
#         self.counter += 1
