import csv

from pax import plugin



class WritePeaksCSV(plugin.OutputPlugin):
    """Write to a human readable CSV file
    """

    def startup(self):
        self.output = open(self.config['output_file'], 'w')
        self.headers = ['event', 'type', 'max', 'left', 'right', 'area']
        self.csv = csv.DictWriter(self.output,
                                  self.headers, lineterminator='\n')
        self.csv.writeheader()


    def write_event(self, event):
        event.peaks.sort(key=lambda p: p.index_of_maximum)

        # # only write down largest S2:
        # # if not event.S2s():
        # #     return
        # # p = event.S2s()[0]
        for p in event.peaks:
            data = (event.event_number,
                    p.type,
                    p.index_of_maximum,
                    p.left,
                    p.right,
                    p.area)
            self.csv.writerow(dict(zip(self.headers, data)))

    def shutdown(self):
        self.output.close()
