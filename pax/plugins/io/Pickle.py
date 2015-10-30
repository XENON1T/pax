"""Read/write event class from/to gzip-compressed pickle files.
"""
import pickle

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
