"""
JSON and BSON-based data output
"""
import json

import bson

from pax import datastructure
from pax.FolderIO import InputFromFolder, WriteToFolder, ReadZipped, WriteZipped


##
# JSON
##
class JSONIO():

    def from_format(self, doc):
        return datastructure.Event(**json.loads(doc))

    def to_format(self, event):
        return event.to_json(fields_to_ignore=self.config['fields_to_ignore'])


class ReadZippedJSON(JSONIO, ReadZipped):
    """Read a folder of zipfiles containing gzipped JSON files"""
    pass


class WriteZippedJSON(JSONIO, WriteZipped):
    """Write raw data to a folder of zipfiles containing gzipped JSONs"""
    pass


class ReadJSON(JSONIO, InputFromFolder):

    """Read raw data from a folder of newline-separated-JSON files
    """
    file_extension = 'json'

    def open(self, filename):
        self.current_file = open(filename, mode='r')

    def get_all_events_in_current_file(self):
        for line in self.current_file:
            yield self.from_format(line)

    def close(self):
        self.current_file.close()


class WriteJSON(JSONIO, WriteToFolder):

    """Write raw data to a folder of newline-separated-JSON files
    """
    file_extension = 'json'

    def open(self, filename):
        self.current_file = open(filename, mode='w')

    def write_event_to_current_file(self, event):
        self.current_file.write(self.to_format(event))
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
