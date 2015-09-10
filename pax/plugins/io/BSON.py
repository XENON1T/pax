"""JSON and BSON-based data output to files
"""
import json
import lzma

import bson

from pax import datastructure
from pax.FolderIO import InputFromFolder, WriteToFolder, ReadZipped, WriteZipped


##
# JSON
##


class ReadJSON(InputFromFolder):

    """Read raw data from a folder of newline-separated-JSON files
    """
    file_extension = 'json'

    def open(self, filename):
        self.current_file = open(filename, mode='r')

    def get_all_events_in_current_file(self):
        for line in self.current_file:
            yield datastructure.Event(**json.loads(line))

    def close(self):
        self.current_file.close()


class WriteJSON(WriteToFolder):

    """Write raw data to a folder of newline-separated-JSON files
    """
    file_extension = 'json'

    def open(self, filename):
        self.current_file = open(filename, mode='w')

    def write_event_to_current_file(self, event):
        self.current_file.write(event.to_json(fields_to_ignore=self.config['fields_to_ignore']))
        self.current_file.write("\n")

    def close(self):
        self.current_file.close()


##
# BSON
##

class BSONIO():

    def from_format(self, doc):
        return datastructure.Event.from_bson(doc)

    def to_format(self, event):
        return event.to_bson(fields_to_ignore=self.config['fields_to_ignore'])


class ReadZippedBSON(BSONIO, ReadZipped):

    """Read a folder of zipfiles containing gzipped BSON files"""
    pass


class WriteZippedBSON(BSONIO, WriteZipped):

    """Write raw data to a folder of zipfiles containing gzipped BSONs"""
    pass


class ReadBSON(InputFromFolder, BSONIO):

    """Read raw BSON data from a concatenated-BSON file or a folder of such files
    """
    file_extension = 'bson'

    def open(self, filename):
        self.current_file = open(filename, mode='rb')
        self.reader = bson.decode_file_iter(self.current_file)

    def close(self):
        self.current_file.close()

    def get_all_events_in_current_file(self):
        for doc in self.reader:
            yield datastructure.Event(**doc)


class WriteBSON(WriteToFolder, BSONIO):

    """Write raw data to a folder of concatenated-BSON files
    """
    file_extension = 'bson'

    def open(self, filename):
        self.current_file = open(filename, mode='wb')

    def write_event_to_current_file(self, event):
        self.current_file.write(self.to_format(event))

    def close(self):
        self.current_file.close()


class LTSReadBSON(InputFromFolder, BSONIO):

    """Long term storage BSON reading

    Read raw BSON data from a concatenated-BSON file or a folder of such files
    """
    file_extension = 'xz'

    def open(self, filename):
        self.current_file = lzma.open(filename,
                                      "r")
        self.reader = bson.decode_file_iter(self.current_file)

    def close(self):
        self.current_file.close()

    def get_all_events_in_current_file(self):
        for doc in self.reader:
            yield datastructure.Event(**doc)


class LTSWriteBSON(WriteToFolder, BSONIO):

    """Long term storage BSON writing

    Write raw data to a folder of concatenated-BSON files.  LZMA and a delta
    compressor are used to shrink the data size down.  A delta compressor just
    stores the bits differences between adjacent waveform samples, which
    compresses well.  This format is intended to be CPU intensive to the benefit
    of disk storage.  Therefore, spare CPU cycles can be used to reduce data.
    """
    file_extension = 'xz'

    def open(self, filename):
        my_filters = [
            {"id": lzma.FILTER_DELTA,
             "dist": self.config['dist']},
            {"id": lzma.FILTER_LZMA2,
             "preset": self.config['preset'] | lzma.PRESET_EXTREME},
        ]
        self.current_file = lzma.open(filename,
                                      "w",
                                      check=lzma.CHECK_SHA256,
                                      filters=my_filters)

    def write_event_to_current_file(self, event):
        self.current_file.write(self.to_format(event))

    def close(self):
        self.current_file.close()
