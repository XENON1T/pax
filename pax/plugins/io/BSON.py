"""
BSON-based raw data output

"""
import bson
import json
import numpy as np
import zipfile
import gzip

import pax      # For version number
from pax import datastructure
from pax.plugins.io.FolderIO import InputFromFolder, WriteToFolder


class ReadBSONBase(InputFromFolder):

    def _doc_to_event(self, doc):
        """Return pax event from dictionary doc
        HACK: we have to cast some stuff to int manually, it appears a newer version of pymongo (3.0?)
        introduces a custom bson int64 class... Why not use numpy int64? How many ints are there??
        """
        pax_event = datastructure.Event(n_channels=self.config['n_channels'],
                                        sample_duration=self.config['sample_duration'],
                                        start_time=int(doc['start_time']),
                                        stop_time=int(doc['stop_time']),
                                        event_number=int(doc['event_number']))

        # For all pulses/pulses, add to pax event
        for pulse in doc['pulses']:

            pulse = datastructure.Pulse(channel=int(pulse['channel']),
                                        left=int(pulse['left']),
                                        raw_data=np.fromstring(pulse['payload'],
                                                               dtype=np.int16))
            pax_event.pulses.append(pulse)

        return pax_event


class ReadBSON(ReadBSONBase):
    """Read raw BSON data from a concatenated-BSON file or a folder of such files
    """
    file_extension = 'bson'

    def open(self, filename):
        self.current_file = open(filename, mode='rb')
        self.reader = bson.decode_file_iter(self.current_file)
        # The first bson is a fake "event" containing metadata, which we ignore:
        next(self.reader)

    def close(self):
        self.current_file.close()

    def get_all_events_in_current_file(self):
        for doc in self.reader:
            yield self._doc_to_event(doc)


class ReadZippedBSON(ReadBSONBase):
    """Read a folder of zipfiles containing gzipped BSON files
    """
    file_extension = 'zip'

    def open(self, filename):
        self.current_file = zipfile.ZipFile(filename)
        self.event_numbers = sorted([int(x)
                                     for x in self.current_file.namelist()
                                     if x != 'metadata'])

    def get_single_event_in_current_file(self, event_position):
        event_name_in_zip = str(self.event_numbers[event_position])
        with self.current_file.open(event_name_in_zip) as event_file_in_zip:
            doc = event_file_in_zip.read()
            doc = gzip.decompress(doc)
            doc = bson.BSON.decode(doc)
            return self._doc_to_event(doc)

    def close(self):
        """Close the currently open file"""
        self.current_file.close()


class WriteBSON(WriteToFolder):

    """Write raw data to a folder of concatenated-BSON files
    """
    file_extension = 'bson'

    def open(self, filename):
        self.current_file = open(filename, mode='wb')
        self._write_metadata()

    def write_event_to_current_file(self, event):
        # Don't use event.to_dict(), this will convert the ndarrays of sample values to lists...
        self._write_doc(name=str(event.event_number),
                        doc=dict(start_time=event.start_time,
                                 stop_time=event.stop_time,
                                 event_number=event.event_number,
                                 pulses=[dict(left=pulse.left,
                                              channel=pulse.channel,
                                              payload=pulse.raw_data.tostring())
                                         for pulse in event.pulses]))

    def close(self):
        self.current_file.close()

    def _write_metadata(self, write_as='bson'):
        self._write_doc(name='metadata',
                        write_as=write_as,
                        doc=dict(run_number=self.config['run_number'],
                                 tpc=self.config['tpc_name'],
                                 file_builder_name='pax',
                                 file_builder_version=pax.__version__))

    def _write_doc(self, doc, name=None, write_as='bson'):
        """Serializes a dictionary to the currently bson file.
        name and write_as are ignored; meant for derivative classes
        """
        self.current_file.write(bson.BSON.encode(doc))


class WriteZippedBSON(WriteBSON):
    """Write raw data to a folder of zipfiles containing gzipped BSONs
    """
    file_extension = 'zip'

    def open(self, filename):
        self.current_file = zipfile.ZipFile(filename, mode='w')
        # Write metadata as json for human readability
        self._write_metadata(write_as='json')

    def _write_doc(self, doc, name=None, write_as='bson'):
        """Serializes a dictionary to the currently open file.
          - name will be name of file in ZipFile
          - write_as specifies format: bson or json
        bsons will be compressed by gzip -- it's faster than the compressions built into ZipFile
        """
        if name is None:
            raise ValueError("Document must be named to go in zip file!")
        if write_as == 'bson':
            to_write = gzip.compress(bson.BSON.encode(doc), self.config['compresslevel'])
        elif write_as == 'json':
            to_write = json.dumps(doc)
        else:
            raise ValueError("Invalid serialization format %s" % write_as)
        self.current_file.writestr(name, to_write)
