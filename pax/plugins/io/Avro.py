"""Avro is responsible for the raw digitizer data storage

Avro is a remote procedure call and data serialization framework developed
within Apache's Hadoop project.  We use it within 'pax' to store the raw data
from the experiment to disk.  These classes are used, for example, by the data
aquisition system to write the raw data coming from the experiment.  More
information about Avro can be found at::

  http://en.wikipedia.org/wiki/Apache_Avro

This replaced 'xdio' from XENON100.
"""
import time

import numpy as np

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import pax      # For version
from pax import plugin, datastructure


class ReadAvro(plugin.InputPlugin):
    """Read raw Avro data to get PMT pulses

    This is the lowest level data stored.
    """

    def startup(self):
        self.reader = DataFileReader(open(self.config['input_name'],
                                          'rb'),
                                     DatumReader())

        # n_channels is needed to initialize pax events.
        self.n_channels = self.config['n_channels']
        self.log.debug("Assuming %d channels",
                       self.n_channels)
        self.log.info(next(self.reader))

    def get_events(self):
        """Fetch events from Avro file

        This produces a generator for all the events within the file.  These
        Events contain occurences, and the appropriate pax objects will be
        built.
        """

        for avro_event in self.reader:  # For every event in file
            # Start the clock
            ts = time.time()

            # Make pax object
            pax_event = datastructure.Event(n_channels=self.n_channels,
                                            start_time=avro_event['start_time'],
                                            stop_time=avro_event['stop_time'],
                                            event_number=avro_event['number'])

            # For all pulses/occurrences, add to pax event
            for pulse in avro_event['pulses']:

                pulse = datastructure.Occurrence(channel=pulse['channel'],
                                                 left=pulse['left'],
                                                 raw_data=np.fromstring(pulse['payload'],
                                                                        dtype=np.int16))
                pax_event.occurrences.append(pulse)

            self.total_time_taken += (time.time() - ts) * 1000

            yield pax_event

    def shutdown(self):
        self.reader.close()


class WriteAvro(plugin.OutputPlugin):

    """Write raw Avro data of PMT pulses

    This is the lowest level data stored.
    """

    def startup(self):
        # The 'schema' stores how the data will be recorded to disk.  This is
        # also saved along with the output.  The schema can be found in
        # _base.ini and outlines what is stored.
        self.schema = avro.schema.Parse(self.config['raw_pulse_schema'])

        self.writer = DataFileWriter(open(self.config['output_name'],
                                          'wb'),
                                     DatumWriter(),
                                     self.schema,
                                     codec=self.config['codec'])

        self.writer.append({'number': -1,
                            'start_time': -1,
                            'stop_time': -1,
                            'pulses': None,
                            'meta': {'run_number': self.config['run_number'],
                                     'tpc': self.config['tpc_name'],
                                     'file_builder_version': pax.__version__
                                     }})

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
