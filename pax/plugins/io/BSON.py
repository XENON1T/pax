"""
JSON and BSON-based data output
"""
import io
import json
import zipfile
import gzip
import pickle

import bson
from rsa import bigfile as rsa_bigfile
from pax import datastructure, utils
from pax.FolderIO import InputFromFolder, WriteToFolder


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

class ReadBSON(InputFromFolder):

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


class WriteBSON(WriteToFolder):

    """Write raw data to a folder of concatenated-BSON files
    """
    file_extension = 'bson'

    def open(self, filename):
        self.current_file = open(filename, mode='wb')

    def write_event_to_current_file(self, event):
        self.current_file.write(event.to_bson(fields_to_ignore=self.config['fields_to_ignore']))

    def close(self):
        self.current_file.close()


##
# Zipped BSON
##

class ReadZippedBSON(InputFromFolder):

    """Read a folder of zipfiles containing gzipped BSON files
    """
    file_extension = 'zip'
    decryption_key = None

    def open(self, filename):
        self.current_file = zipfile.ZipFile(filename)
        self.event_numbers = sorted([int(x)
                                     for x in self.current_file.namelist()])
        if self.config['decryption_key_file']:
            with open(utils.data_file_name(self.config['decryption_key_file']), 'rb') as f:
                self.decryption_key = pickle.load(f)

    def get_event_numbers_in_current_file(self):
        return self.event_numbers

    def get_single_event_in_current_file(self, event_number):
        with self.current_file.open(str(event_number)) as event_file_in_zip:
            doc = event_file_in_zip.read()

            if self.decryption_key is not None:
                # Open fake binary files to interface with rsa module
                infile = io.BytesIO(doc)
                outfile = io.BytesIO()
                rsa_bigfile.decrypt_bigfile(infile, outfile, self.decryption_key)
                doc = outfile.getvalue()

            doc = gzip.decompress(doc)

            return datastructure.Event.from_bson(doc)

    def close(self):
        """Close the currently open file"""
        self.current_file.close()


class WriteZippedBSON(WriteToFolder):

    """Write raw data to a folder of zipfiles containing gzipped BSONs
    """
    file_extension = 'zip'
    encryption_key = None

    def open(self, filename):
        self.current_file = zipfile.ZipFile(filename, mode='w')
        if self.config['encryption_key_file']:
            with open(utils.data_file_name(self.config['encryption_key_file']), 'rb') as f:
                self.encryption_key = pickle.load(f)

    def write_event_to_current_file(self, event):
        event_data = gzip.compress(event.to_bson(fields_to_ignore=self.config['fields_to_ignore']),
                                   self.config['compresslevel'])

        # If an encryption key is specified
        if self.encryption_key is not None:
            # Open fake binary files to interface with rsa module
            infile = io.BytesIO(event_data)
            outfile = io.BytesIO()
            rsa_bigfile.encrypt_bigfile(infile, outfile, self.encryption_key)
            event_data = outfile.getvalue()

        self.current_file.writestr(str(event.event_number), event_data)

    def close(self):
        self.current_file.close()
