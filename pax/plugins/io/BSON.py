"""JSON and BSON-based data output to files
"""
import json

from pax import datastructure
from pax.FolderIO import InputFromFolder, WriteToFolder, WriteZippedEncoder, ReadZippedDecoder


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
# Zipped BSON
##

class EncodeZBSON(WriteZippedEncoder):

    def encode_event(self, event):
        return event.to_bson(fields_to_ignore=self.config['fields_to_ignore'])


class DecodeZBSON(ReadZippedDecoder):

    def decode_event(self, event):
        event = datastructure.Event.from_bson(event)
        return event
