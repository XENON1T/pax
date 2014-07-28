import csv

from pax import plugin


def is_s2(peak):
    return peak['peak_type'] in ('large_s2', 'small_s2')


class WriteCSVPeakwise(plugin.OutputPlugin):
    def startup(self):
        self.counter = 0

    def write_event(self, event):
        event = event._raw
        for p in event['peaks']:
            data = {
                'event': self.counter,  # TODO: get from mongo/xed/whatever
                'left': p['left'],
                'right': p['right'],
                'area': p['top_and_bottom']['area'],
                'type': p['peak_type'],
            }
            if not hasattr(self, 'csv'):
                self.output = open('output.csv', 'w')
                self.headers = ['event', 'type', 'left', 'right', 'area']  # Grmpfh, needed for order
                self.csv = csv.DictWriter(self.output, self.headers, lineterminator='\n')
                self.csv.writeheader()
            self.csv.writerow(data)
        self.counter += 1


class WriteEventsToCSV(plugin.OutputPlugin):
    def startup(self):
        self.counter = -1
        self.output = open('output.csv', 'w')
        self.headers = ['event', 'largest_s2_area', 'largest_s2_left', 'largest_s2_right']  # Grmpfh, needed for order
        self.csv = csv.DictWriter(self.output, self.headers, lineterminator='\n')
        self.csv.writeheader()

    def write_event(self, event):
        self.counter += 1

        # Find largest s2...
        s2areas = [p['top_and_bottom']['area'] for p in event['peaks'] if is_s2(p)]
        if s2areas == []:
            # No S2s in this waveform - skip event
            return
        # DANGER ugly code ahead...
        s2maxarea = max(s2areas)
        for i, p in enumerate(event['peaks']):
            if p['top_and_bottom']['area'] == s2maxarea:
                largests2 = i

        self.csv.writerow({
            'event': self.counter,
            'largest_s2_area': event['peaks'][largests2]['top_and_bottom']['area'],
            'largest_s2_left': event['peaks'][largests2]['left'],
            'largest_s2_right': event['peaks'][largests2]['right']
        })
        self.counter += 1
