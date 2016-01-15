"""Read/write event class from/to gzip-compressed pickle files.
"""
import pickle

from pax.FolderIO import WriteZippedEncoder, ReadZippedDecoder


##
# Zipped pickles
##

class EncodePickle(WriteZippedEncoder):

    def encode_event(self, event):
        return pickle.dumps(event)


class DecodePickle(ReadZippedDecoder):

    def decode_event(self, event):
        return pickle.loads(event)
