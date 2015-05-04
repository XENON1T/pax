"""
BSON-based raw data output

"""
import bson
import numpy as np

import pax      # For version number
from pax import datastructure
from pax.plugins.io.FolderIO import InputFromFolder, WriteToFolder


class ReadBSON(InputFromFolder):
    """Read raw BSON data from a BSON file or a folder of BSON files
    """
    file_extension = 'bson'

    def open(self, filename):
        self.current_file = open(filename, mode='rb')
        self.reader = bson.decode_file_iter(self.current_file)
        next(self.reader)   # The first datum is a fake "event" containing metadata

    def close(self):
        """Close the currently open file"""
        self.current_file.close()

    def get_all_events_in_current_file(self):
        """Yield events from BSON file iteratively
        """
        for doc in self.reader:  # For every event in file

            # Make pax object
            pax_event = datastructure.Event(n_channels=self.config['n_channels'],
                                            sample_duration=self.config['sample_duration'],
                                            start_time=doc['start_time'],
                                            stop_time=doc['stop_time'],
                                            event_number=doc['event_number'])

            # For all pulses/pulses, add to pax event
            for pulse in doc['pulses']:

                pulse = datastructure.Pulse(channel=pulse['channel'],
                                            left=pulse['left'],
                                            raw_data=np.fromstring(pulse['payload'],
                                                                   dtype=np.int16))
                pax_event.pulses.append(pulse)

            yield pax_event


class WriteBSON(WriteToFolder):

    """Write raw data of PMT pulses to a folder of small BSON files
    """
    file_extension = 'bson'

    def open(self, filename):
        self.current_file = open(filename, mode='wb')

        # Store the metadata as a "first event"
        self._write(dict(run_number=self.config['run_number'],
                         tpc=self.config['tpc_name'],
                         file_builder_name='pax',
                         file_builder_version=pax.__version__))

    def write_event_to_current_file(self, event):
        # Don't use event.to_dict(), this will convert the ndarrays of sample values to lists...
        self._write(dict(start_time=event.start_time,
                         stop_time=event.stop_time,
                         event_number=event.event_number,
                         pulses=[dict(left=pulse.left,
                                      channel=pulse.channel,
                                      payload=pulse.raw_data.tostring())
                                 for pulse in event.pulses]))

    def close(self):
        self.current_file.close()

    def _write(self, doc):
        """Writes a dictionary doc to the currently open bson file"""
        self.current_file.write(bson.BSON.encode(doc))
