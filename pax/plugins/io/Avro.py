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
import os

import numpy as np

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter

import pax      # For version number
from pax import plugin, datastructure
from pax.plugins.io.InputFromFolder import InputFromFolder


class ReadAvro(InputFromFolder):
    """Read raw Avro data from an Avro file or folder of Avro files
    """
    file_extension = 'avro'

    def get_first_and_last_event_number(self, filename):
        _, _, first_event, last_event = os.path.splitext(filename)[0].split('-')
        return int(first_event), int(last_event)

    def start_to_read_file(self, filename):
        self.reader = DataFileReader(open(filename, 'rb'),
                                     DatumReader())
        next(self.reader)   # Skips the metadata, which is in the first event

    def close_current_file(self):
        """Close the currently open file"""
        self.reader.close()

    def get_all_events_in_current_file(self):
        """Yield events from Avro file as  iteratively
        """
        for avro_event in self.reader:  # For every event in file
            # Start the clock
            ts = time.time()

            # Make pax object
            pax_event = datastructure.Event(n_channels=self.config['n_channels'],
                                            sample_duration=self.config['sample_duration'],
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


class WriteAvro(plugin.OutputPlugin):

    """Write raw Avro data of PMT pulses to a folder of small Avro files
    """

    def startup(self):

        # The 'schema' stores how the data will be recorded to disk.  This is
        # also saved along with the output.  The schema can be found in
        # _base.ini and outlines what is stored.
        self.schema = avro.schema.Parse(self.config['event_schema'])

        self.events_per_file = self.config['events_per_file']
        self.first_event_in_current_file = None
        self.last_event_written = None

        self.output_dir = self.config['output_name']
        if os.path.exists(self.output_dir):
            raise ValueError("Output directory %s already exists, can't write your avros there!" % self.output_dir)
        else:
            os.mkdir(self.output_dir)

        self.tempfile = os.path.join(self.output_dir, 'temp.avro')

        # TODO: write the metadata to a separate file

    def write_event(self, event):
        """Write one more event to the avro folder, opening/closing files as needed"""
        if self.last_event_written is None \
                or self.events_written_to_current_file >= self.events_per_file:
            self.open_new_file(first_event_number=event.event_number)

        self.writer.append(dict(number=event.event_number,
                                start_time=event.start_time,
                                stop_time=event.stop_time,
                                pulses=[dict(payload=pulse.raw_data.tobytes(),
                                             left=pulse.left,
                                             channel=pulse.channel)
                                        for pulse in event.occurrences]))

        self.events_written_to_current_file += 1
        self.last_event_written = event.event_number

    def shutdown(self):
        self.close_current_file()

    def open_new_file(self, first_event_number):
        """Opens a new file, closing any old open ones"""
        if self.last_event_written is not None:
            self.close_current_file()
        self.first_event_in_current_file = first_event_number
        self.events_written_to_current_file = 0
        self.writer = DataFileWriter(open(self.tempfile, 'wb'),
                                     DatumWriter(),
                                     self.schema,
                                     codec=self.config['codec'])

        # Store the metadata as a "first event"
        self.writer.append(dict(number=-1,
                                start_time=-1,
                                stop_time=-1,
                                pulses=None,
                                meta=dict(run_number=self.config['run_number'],
                                          tpc=self.config['tpc_name'],
                                          file_builder_name='pax',
                                          file_builder_version=pax.__version__)))

    def close_current_file(self):
        """Closes the currently open file, if there is one"""
        if self.last_event_written is None:
            self.log.info("You didn't write any events... Did you crash pax?")
            return

        self.log.info("Closing current avro file, you'll get a silly 'info' from avro now...")
        self.writer.close()

        # Rename the temporary file to reflect the events we've written to it
        os.rename(self.tempfile,
                  os.path.join(self.output_dir,
                               '%s-%d-%06d-%06d.avro' % (self.config['tpc_name'],
                                                         self.config['run_number'],
                                                         self.first_event_in_current_file,
                                                         self.last_event_written)))
