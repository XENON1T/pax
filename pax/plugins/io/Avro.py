"""Avro is responsible for the raw digitizer data storage

Avro is a remote procedure call and data serialization framework developed
within Apache's Hadoop project.  We use it within 'pax' to store the raw data
from the experiment to disk.  These classes are used, for example, by the data
aquisition system to write the raw data coming from the experiment.  More
information about Avro can be found at::

  http://en.wikipedia.org/wiki/Apache_Avro

This replaced 'xdio' from XENON100.
"""

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
from pax import plugin
import numpy as np
from pax import datastructure

class ReadAvro(plugin.InputPlugin):
    """Read raw Avro data to get PMT pulses

    This is the lowest level data stored.
    """
    def startup(self):
        self.reader = DataFileReader(open(self.config['input_name'],
                                          'rb'),
                                     DatumReader())

        self.n_channels = self.config['n_channels']
        self.log.error("Assuming %d channels" % self.n_channels)

    def get_events(self):
        """Generator of events from Mongo
        """
        for avro_event in self.reader:
            self.log.error(avro_event.keys())
            pax_event = datastructure.Event(n_channels=self.n_channels,
                                            start_time=avro_event['start_time'],
                                            stop_time=avro_event['stop_time'],
                                            event_number=avro_event['number'])

            for pulse in avro_event['pulses']:

                pulse = datastructure.Occurrence(channel=pulse['channel'],
                                                 left=pulse['left'],
                                                 raw_data=np.fromstring(pulse['payload'],
                                                                        dtype=np.int16))
                pax_event.occurrences.append(pulse)

            yield pax_event

    def shutdown(self):
        self.reader.close()

class WriteAvro(plugin.OutputPlugin):
    """Write raw Avro data of PMT pulses

    This is the lowest level data stored.
    """
    def startup(self):
        #  The 'schema' stores how the data will be recorded to disk.  This is
        #  also saved along with the output.
        self.schema = avro.schema.Parse(self.config['raw_pulse_schema'])

        self.writer = DataFileWriter(open(self.config['output_name'],
                                          'wb'),
                                     DatumWriter(),
                                     self.schema)

    def write_event(self, pax_event):
        self.log.debug('Writing event')
        avro_event = {}
        avro_event['number'] = pax_event.event_number
        avro_event['start_time'] = pax_event.start_time
        avro_event['stop_time'] = pax_event.stop_time
        avro_event['pulses'] = []

        for pax_pulse in pax_event.occurrences:
            avro_pulse = {}

            avro_pulse['payload'] = pax_pulse.raw_data.tobytes()
            avro_pulse['left'] = pax_pulse.left
            avro_pulse['channel'] = pax_pulse.channel

            avro_event['pulses'].append(avro_pulse)

        self.writer.append(avro_event)

    def shutdown(self):
        self.writer.close()


