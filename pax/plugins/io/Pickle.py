"""Read/write event class from/to gzip-compressed pickle files.
"""

import gzip
import glob
import os
import pickle
import re

from pax import plugin
from pax.FolderIO import ReadZipped, WriteZipped


##
# Zipped pickles
##


class PickleIO():

    def from_format(self, doc):
        return pickle.loads(doc)

    def to_format(self, event):
        return pickle.dumps(event)


class ReadZippedPickles(PickleIO, ReadZipped):
    """Read a folder of zipfiles containing gzipped pickle files"""
    pass


class WriteZippedPickles(PickleIO, WriteZipped):
    """Write raw data to a folder of zipfiles containing gzipped pickles"""
    pass


##
# Single events to pickles
# TOD: This is quite old code, we're keeping it for compatibility with pftest only
##

class WriteToPickleFile(plugin.OutputPlugin):

    def write_event(self, event):
        output_dir = self.config['output_name']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.log.debug("Starting pickling...")
        with gzip.open(os.path.join(output_dir, '%06d' % event.event_number),
                       'wb', compresslevel=1) as file:
            pickle.dump(event, file)
        self.log.debug("Done!")


class DirWithPickleFiles(plugin.InputPlugin):

    def startup(self):
        files = glob.glob(self.config['input_name'] + "/*")
        self.event_files = {}
        if len(files) == 0:
            self.log.fatal("No files found in input directory %s!" % self.config['input_name'])
        for file in files:
            m = re.search('(\d+)$', file)
            if m is None:
                self.log.debug("Invalid file %s" % file)
                continue
            else:
                self.event_files[int(m.group(0))] = file
        if len(self.event_files) == 0:
            self.log.fatal("No valid event files found in input directory %s!" % self.config['input_name'])
        self.number_of_events = len(self.event_files)

    def get_single_event(self, index):
        file = self.event_files[index]
        with gzip.open(file, 'rb') as f:
            return pickle.load(f)

    def get_events(self):
        for index in sorted(self.event_files.keys()):
            yield self.get_single_event(index)
