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
                self.csv = csv.DictWriter(
                    self.output, self.headers, lineterminator='\n')
                self.csv.writeheader()
            self.csv.writerow({a[0]: a[1] for a in data})

    def shutdown(self):
        self.output.close()
