"""(deprecated for BSON) Avro is responsible for the raw digitizer data storage

Avro is a remote procedure call and data serialization framework developed
within Apache's Hadoop project.  We use it within 'pax' to store the raw data
from the experiment to disk.  These classes are used, for example, by the data
aquisition system to write the raw data coming from the experiment.  More
information about Avro can be found at::

  http://en.wikipedia.org/wiki/Apache_Avro

This replaced 'xdio' from XENON100.
"""
import gzip
from io import BytesIO

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import numpy as np

import pax      # For version number
from pax import datastructure
from pax.plugins.io.FolderIO import InputFromFolder, WriteToFolder


class ReadAvro(InputFromFolder):
    """Read raw Avro data from an Avro file or folder of Avro files
    """
    file_extension = 'avro'

    def open(self, filename):

        # Gzip is a stream compression algorithm, while Avro requires seeking
        # (which makes its lack of random access support ever weirder...)
        # so we need to uncompress the entire file into memory
        # (BytesIO is a 'fake in-memory binary file')
        with gzip.open(filename, mode='rb') as infile:
            f = BytesIO(infile.read())

        self.reader = DataFileReader(f, DatumReader())
        next(self.reader)   # The first datum is a fake "event" containing metadata

    def close(self):
        """Close the currently open file"""
        self.reader.close()

    def get_all_events_in_current_file(self):
        """Yield events from Avro file iteratively
        """
        for avro_event in self.reader:  # For every event in file

            # Make pax object
            pax_event = datastructure.Event(n_channels=self.config['n_channels'],
                                            sample_duration=self.config['sample_duration'],
                                            start_time=avro_event['start_time'],
                                            stop_time=avro_event['stop_time'],
                                            event_number=avro_event['number'])

            # For all pulses/pulses, add to pax event
            for pulse in avro_event['pulses']:

                pulse = datastructure.Pulse(channel=pulse['channel'],
                                            left=pulse['left'],
                                            raw_data=np.fromstring(pulse['payload'],
                                                                   dtype=np.int16))
                pax_event.pulses.append(pulse)

            yield pax_event


class WriteAvro(WriteToFolder):

    """Write raw Avro data of PMT pulses to a folder of small Avro files
    """
    file_extension = 'avro'

    def startup(self):
        # The 'schema' stores how the data will be recorded to disk.  This is
        # also saved along with the output.  The schema can be found in
        # _base.ini and outlines what is stored.
        self.schema = avro.schema.Parse(self.config['event_schema'])
        super().startup()

    def open(self, filename):
        self.log.warning("Avro is no longer supported and will "
                         "be removed soon. Use BSON.")
        # Open file and set codec
        f = gzip.open(filename, mode='wb', compresslevel=self.config.get('compresslevel', 4))

        # Start avro writer
        self.writer = DataFileWriter(f, DatumWriter(), self.schema, codec='null')

        # Store the metadata as a "first event"
        self.writer.append(dict(number=-1,
                                start_time=-1,
                                stop_time=-1,
                                pulses=None,
                                meta=dict(run_number=self.config['run_number'],
                                          tpc=self.config['tpc_name'],
                                          file_builder_name='pax',
                                          file_builder_version=pax.__version__)))

    def write_event_to_current_file(self, event):
        self.writer.append(dict(number=event.event_number,
                                start_time=event.start_time,
                                stop_time=event.stop_time,
                                pulses=[dict(payload=pulse.raw_data.tobytes(),
                                             left=pulse.left,
                                             channel=pulse.channel)
                                        for pulse in event.pulses]))

    def close(self):
        self.log.info("Closing current avro file, you'll get a silly 'info' from avro now...")
        self.writer.close()
